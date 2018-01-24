from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
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

def download_data():
    print(os.getcwd())
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
    _, word2index_map = load_vocab(vocab_path)
    embedding_weights = {}
    count_all_words = 0 
    embed = []
    with ZipFile(path_to_glove) as z:
        with z.open("glove.6B.100d.txt") as f:
            print('get glove word vector!')
            for line in tqdm(f):
                vals = line.split()
                word = str(vals[0].decode("utf-8")) 
                if word in word2index_map:
                    count_all_words+=1
                    coefs = np.asarray(vals[1:], dtype='float32')
                    coefs /= np.linalg.norm(coefs)
                    embedding_weights[word] = coefs
                    embed.append(coefs)
                if count_all_words == config.VOCAB_SIZE:
                        break
        embedding_matrix = np.zeros((config.VOCAB_SIZE, config.GLOVE_SIZE))
        print('filling embedding matrix')
        for word, index in tqdm(word2index_map.items()):
            if not word == "<pad>":
                try:
                    word_embedding = embedding_weights[word]
                    embedding_matrix[index, :] = word_embedding
                except:
                    pass
    return embedding_matrix


def sentence_tokenizer(sentence, stem=False, stopword=True, normalize_numbers=True):
    sentence = sentence.lower()
    if normalize_numbers:
        replace_numbers = re.compile(r'\d+',re.IGNORECASE)
        sentence = replace_numbers.sub('number', sentence)
    tokenizer = RegexpTokenizer(r'\w+')
    sentence = tokenizer.tokenize(sentence)
    sentence = [word for word in sentence if word not in string.punctuation]
    sentence = [word for word in sentence if len(word)>1]
    if stopword:
        sentence = [word for word in sentence if not word in stopwords.words('english')]
    if stem:
        porter_stemmer = PorterStemmer()
        sentence = [porter_stemmer.stem(word) for word in sentence]

    return sentence
        
        
        
def build_vocab(filename):
    in_path = os.path.join(config.DATA_PATH, filename)
    out_path = os.path.join(config.PROCESSED_PATH, 'vocab.txt')
    
    if config.PROCESSED_PATH not in os.listdir(config.DATA_PATH):
        try:
            os.mkdir(config.PROCESSED_PATH)
        except OSError:
            pass

    vocab = {}
    
    df = read_file(in_path)
    print('tokenizing vocab file')
    for line in tqdm(df.comment_text):
        for token in sentence_tokenizer(line):
            if token not in vocab:
                vocab[token] = 0
            vocab[token] += 1
        
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with open(out_path, 'w') as f:
        f.write('<pad>' + '\n')
        f.write('<unk>' + '\n')
        # f.write('<s>' + '\n')
        # f.write('<\s>' + '\n') 
        index = 4
        print('writing vocab file')
        for word in tqdm(sorted_vocab):
            if vocab[word] < config.THRESHOLD:
                with open('src/config.py', 'a') as cf:
                        cf.write('VOCAB_SIZE = ' + str(index) + '\n')
                break
            f.write(word + '\n')
            index += 1
            
            
def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}
    


def sentence2id(vocab, line):
    # return [vocab.get(token, vocab['<unk>']) for token in sentence_tokenizer(line)]
    return [vocab.get(token, vocab['<unk>'] ) for token in sentence_tokenizer(line)]



def token2id(filename, out_path='train_ids.txt'):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab.txt'
    _, vocab = load_vocab(os.path.join(config.PROCESSED_PATH, vocab_path))
    in_file = os.path.join(config.DATA_PATH, filename)
    df = read_file(in_file)
    out_file = open(os.path.join(config.PROCESSED_PATH, out_path), 'w')
    
    lines = df.comment_text.fillna('NA').values
    print('token to ids')
    for line in tqdm(lines):
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

def _get_buckets():
    """ Load the dataset into buckets based on their lengths.
    train_buckets_scale is the inverval that'll help us 
    choose a random bucket later on.
    """
    # test_buckets = data.load_data('test_ids.enc')
    data_buckets = load_data('train_ids.txt')
    train_bucket_sizes = [len(data_buckets[b]) for b in range(len(config.BUCKETS))]
    print("Number of samples in each bucket:\n", train_bucket_sizes)
    train_total_size = sum(train_bucket_sizes)
    # list of increasing numbers from 0 to 1 that we'll use to select a bucket.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    print("Bucket scale:\n", train_buckets_scale)
    return data_buckets, train_buckets_scale


def process_data():
    download_data()
    print('Preparing data to be model-ready ...')
    build_vocab('train.csv')
    token2id('train.csv')
    token2id('test.csv', 'test_ids.txt')


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


if __name__ == '__main__':
    process_data()
