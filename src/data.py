from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.stem.wordnet import WordNetLemmatizer

import time
import re
import pandas as pd
from zipfile import ZipFile
import os
import string
import config
import random
from urllib.request import urlretrieve 
import numpy as np
from tqdm import tqdm
import gc 

from gensim.models.phrases import Phraser, Phrases
from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary
from textacy import preprocess
import multiprocessing
import itertools


replace_numbers = re.compile(r'\d+',re.IGNORECASE)
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
tokenize = RegexpTokenizer(r'\w+')
alpha_numeric = re.compile('[\W_]+')

with open(os.path.join(config.DATA_PATH, 'bad_words_en.txt'), 'r') as f:
    bad_words = f.read().split('\n')[:-1]

def substitute_repeats_fixed_len(text, num_chars, num_times=3):
    return re.sub(r"(\S{{{}}})(\1{{{},}})".format(num_chars, num_times-1), r"\1", text)

def substitute_repeats(text, num_times=3):
    for num_chars in range(1, 20):
        text = substitute_repeats_fixed_len(text, num_chars, num_times)
    return text

def split_text_and_digits(text, regexps=None):
    if not regexps:
        regexps = [re.compile("([a-zA-Z]+)([0-9]+)"),
                   re.compile("([0-9]+)([a-zA-Z]+)")]
    for regexp in regexps:
        result = regexp.match(text)
        if result is not None:
            return ' '.join(result.groups())
    return text

def is_bad(token):
    token = token.lower()
    bads = set()
    for word in bad_words:
        if word in token:
            if word == token:
                bads.add(token)
            else:
                bads.add(token)
                bads.add(word)
    
    if bads:
        return ' '.join(bads)
    else:
        return token

def download_data():
    train_url = 'https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/download/train.csv.zip'
    test_url = 'https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/download/test.csv.zip'
    glove_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    if not os.path.isfile('./data/train.csv.zip'):
        print('download training data')
        urlretrieve(train_url, filename='./data/train.csv.zip')
    if not os.path.isfile('./data/test.csv.zip'):
        print('download test data')
        urlretrieve(test_url, filename='./data/test.csv.zip')
    if not os.path.isfile('./data/glove.6B.zip'):
        print('download glove embeddings')
        urlretrieve(glove_url, filename='./data/glove.6B.zip')


def read_file(filename):
    with ZipFile(filename + '.zip') as myzip:
        with myzip.open(myzip.filelist.pop()) as myfile:
            df = pd.read_csv(myfile)
    return df


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def get_glove(path_to_glove, vocab_path):
    vocab = load_vocab(vocab_path)
    embedding_weights = {}
    count_all_words = 0 
    embed = []
    with ZipFile(path_to_glove) as z:
        with z.open("glove.6B.100d.txt") as f:
            print('get glove word vector!')
            for line in tqdm(f):
                vals = line.split()
                word = str(vals[0].decode("utf-8")) 
                if word in vocab.itervalues():
                    count_all_words+=1
                    coefs = np.asarray(vals[1:], dtype='float32')
                    coefs /= np.linalg.norm(coefs)
                    embedding_weights[word] = coefs
                    embed.append(coefs)
                if count_all_words == config.VOCAB_SIZE:
                        break
        embedding_matrix = np.zeros((config.VOCAB_SIZE, config.GLOVE_SIZE))
        print('filling embedding matrix')
        for index, word in tqdm(vocab.items(), total=config.VOCAB_SIZE):
            if not word == "<pad>":
                try:
                    word_embedding = embedding_weights[word]
                    embedding_matrix[index, :] = word_embedding
                except:
                    pass
    return embedding_matrix


def parallelize_dataframe(comments, func):
    num_cores = multiprocessing.cpu_count() - 1  #leave one free to not freeze machine
    num_partitions = num_cores #number of partitions to split dataframe
    comments_split = np.array_split(comments, num_partitions)
    pool = multiprocessing.Pool(num_cores)
    output = np.concatenate(pool.map(func, comments_split))
    pool.close()
    pool.join()
    return output


def my_preprocess(sentence, stopword=1, stem=0, lemma=1):
    sentence = alpha_numeric.sub(' ', sentence)
    sentence = replace_numbers.sub(' number ', sentence)
    sentence = tokenize.tokenize(sentence)
    sentence = [split_text_and_digits(token) for token in sentence]
    sentence = [substitute_repeats(token, 3) for token in sentence]
    sentence = [is_bad(token) for token in sentence]
    
    sentence = [word for word in sentence if len(word) > 1]
    if stopword:
        sentence = [word for word in sentence if not word in stopwords.words('english')]
    if stem:
        sentence = [porter_stemmer.stem(word) for word in sentence]
    if lemma: 
        sentence = [wordnet_lemmatizer.lemmatize(word) for word in sentence]

    return ' '.join(sentence)

def tokenizer(sentences):
    y = []
    if type(sentences) == str:
        sentences = [sentences]
    for comment in sentences:
        comment = my_preprocess(comment)
        txt = preprocess.normalize_whitespace(comment)
        
        txt = preprocess.preprocess_text(txt, 
                                         fix_unicode=True, 
                                         lowercase=True, 
                                         transliterate=True, 
                                         no_urls=True, 
                                         no_emails=True,
                                         no_phone_numbers=True,
                                         no_numbers=True,
                                         no_currency_symbols=True, 
                                         no_punct=True,
                                         no_contractions=True,
                                         no_accents=True)

        y.append(u''.join(txt))
    return y
        
def build_vocab():
    start = time.time()
    test_path = os.path.join(config.DATA_PATH, 'test.csv')
    train_path = os.path.join(config.DATA_PATH, 'train.csv')
    normalized_text_path = os.path.join(config.PROCESSED_PATH, 'normalized_comments.txt')
    bigram_path = os.path.join(config.PROCESSED_PATH, 'bigram')
    bigram_comments_path = os.path.join(config.PROCESSED_PATH, 'bigram_commnets.txt')
    
    if config.PROCESSED_PATH not in os.listdir(config.DATA_PATH):
        try:
            os.mkdir(config.PROCESSED_PATH)
        except OSError:
            pass

    vocab = {}
    
    train_df = read_file(train_path)
    test_df = read_file(test_path)
    print('tokenizing vocab file')
    texts =  np.concatenate([train_df.comment_text.fillna('N/A').values,
                             test_df.comment_text.fillna('N/A').values])


    with open(normalized_text_path, 'w') as f:
        processed_text = parallelize_dataframe(texts, tokenizer)
        for line in processed_text:
            f.write(line + '\n')
    gc.collect()
    lines = LineSentence(normalized_text_path)
    bigram = Phrases(lines)
    bigram.save(bigram_path)
    phraser = Phraser(bigram)

    with open(bigram_comments_path, 'w', encoding='utf_8') as f: 
       for comment in lines:
            comm = u' '.join(phraser[comment])
            f.write(comm + '\n')

    commnets = LineSentence(bigram_comments_path)
    bigram_dict = Dictionary(commnets)
    bigram_dict.filter_extremes(no_below=config.THRESHOLD)
    bigram_dict.save_as_text(config.VOCAB_PATH)
    bigram_dict.add_documents([['<pad>']])

    with open(os.path.join(config.ROOT, 'src', 'config.py'), 'a') as f:
        f.write('VOCAB_SIZE = {}'.format(len(bigram_dict)))    

    print('time passed: {} minutes'.format((time.time() - start) / 60))   


def load_vocab(vocab_path):
    dict = Dictionary.load_from_text(vocab_path)
    return dict
    

def sentence2id(vocab, tokens):
    return vocab.doc2idx(tokens)


def token2id(filename, out_path):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab.txt'
    vocab = load_vocab(os.path.join(config.PROCESSED_PATH, vocab_path))
    in_file = os.path.join(config.DATA_PATH, filename)
    df = read_file(in_file)
    out_file = open(os.path.join(config.PROCESSED_PATH, out_path), 'w')
    
    lines = df.comment_text.fillna('N/A').values
    print('token to ids')
    for line in tqdm(lines):
        line = tokenizer(line)[0].split()
        ids = sentence2id(vocab, line)
        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')
        


def train_test_split(inputs, targets, train_ratio):
    indx = list(range(len(inputs)))
    random.shuffle(indx)
    train_length = int(len(inputs) * train_ratio)

    train_input = [inputs[i] for i in indx[:train_length]]
    train_target = [targets[i] for i in indx[:train_length]]

    test_input = [inputs[i] for i in indx[train_length:]]
    test_target = [targets[i] for i in indx[train_length:]]

    return train_input, train_target, test_input, test_target

def load_data(text_id_filename, target_file_name, mode='train'):
    with open(os.path.join(config.PROCESSED_PATH, text_id_filename), 'r') as file:
        print('loading {}'.format(text_id_filename))
        data = []
        lines = file.readlines()
        for line in tqdm(lines):
            ids = [int(id_) for id_ in line.split()]
            data.append(ids)
    
    if mode == 'train':
        in_file = os.path.join(config.DATA_PATH, target_file_name)
        df = read_file(in_file)
        targets = df.iloc[:, 2:].values

        train_input, train_target, test_input, test_target = train_test_split(data, targets, config.TRAIN_TEST_RATIO)
    
        return train_input, train_target, test_input, test_target
    
    if mode == 'test':
        in_file = os.path.join(config.DATA_PATH, target_file_name)
        df = read_file(in_file)
        ids = df.iloc[:, 0]
        return data, ids 


def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                    for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs


def _pad_input(input_, size=config.MAX_SEQ_LENGTH):
    if len(input_) > config.MAX_SEQ_LENGTH:
        output = input_[:config.MAX_SEQ_LENGTH]
    else:
        output = input_ + [config.PAD_ID] * (size - len(input_))

    return output

def get_batch(data, target, batch_size=None):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket
    if not batch_size:
        batch_size = len(data)

    indx = list(range(len(data)))
    random.shuffle(indx)

    batch_data = [data[i] for i in indx[:batch_size]]
    if target:
        assert len(data) == len(target)
        batch_target = [target[i] for i in indx[:batch_size]]
    inputs = [_pad_input(line) for line in batch_data]
    inputs_length = [len(line) if len(line) < config.MAX_SEQ_LENGTH else config.MAX_SEQ_LENGTH for line in batch_data]
    return inputs, batch_target, inputs_length


def get_test_data(data, ids):
    inputs = []
    inputs_length = []

    for line in data:
        if len(line) == 0:
            line = [1]
        pad_seq = list(_pad_input(line, config.MAX_SEQ_LENGTH))
        inputs.append(pad_seq)
        inputs_length.append(len(pad_seq))

    return inputs, ids, inputs_length

def process_data():
    download_data()
    print('Preparing data to be model-ready ...')
    build_vocab()
    token2id('train.csv', 'train_ids.txt')
    token2id('test.csv', 'test_ids.txt')


if __name__ == '__main__':
    process_data()
