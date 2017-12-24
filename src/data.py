from nltk.tokenize import word_tokenize
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


def download_data():
    print(os.getcwd())
    train_url = 'https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/download/train.csv.zip'
    test_url = 'https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/download/test.csv.zip'
    if not os.path.isfile('./data/train.csv.zip'):
        urlretrieve(train_url, filename='./data/train.csv.zip')
    if not os.path.isfile('./data/test.csv.zip'):
        urlretrieve(test_url, filename='./data/test.csv.zip')


def read_file(filename):
    with ZipFile(filename) as myzip:
        with myzip.open('train.csv') as myfile:
            df = pd.read_csv(myfile)
    return df

def sentence_tokenizer(sentence, stem=False, stopwords=False):
    sentence = sentence.lower()
    sentence = word_tokenize(sentence)
    sentence = [word for word in sentence if word not in string.punctuation]
    if stopwords:
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
    for line in df.comment_text:
        for token in sentence_tokenizer(line):
            if token not in vocab:
                vocab[token] = 0
            vocab[token] += 1
        
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with open(out_path, 'w') as f:
        f.write('<pad>' + '\n')
        f.write('<unk>' + '\n')
        f.write('<s>' + '\n')
        f.write('<\s>' + '\n') 
        index = 4
        for word in sorted_vocab:
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
    return [vocab.get(token, vocab['<unk>']) for token in sentence_tokenizer(line)]


def token2id(filename):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab.txt'
    _, vocab = load_vocab(os.path.join(config.PROCESSED_PATH, vocab_path))
    in_file = os.path.join(config.DATA_PATH, filename)
    df = read_file(in_file)
    out_path = 'train_ids.txt' 
    out_file = open(os.path.join(config.PROCESSED_PATH, out_path), 'w')
    
    lines = df.comment_text.values
    for line in lines:
        ids = sentence2id(vocab, line)
        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')
        
        
def load_data(filename, max_training_size=None):
    file = open(os.path.join(config.PROCESSED_PATH, filename), 'r')
    file = file.readline()
    data_buckets = [[] for _ in config.BUCKETS]
    i = 0
    while file:
        if (i + 1) % 10000 == 0:
            print("Bucketing conversation number", i)
        ids = [int(id_) for id_ in file.split()]
        for bucket_id, encode_max_size in enumerate(config.BUCKETS):
            if len(ids) <= encode_max_size:
                data_buckets[bucket_id].append(ids)
                break
        file = file.readline()
        i += 1
    return data_buckets


def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                    for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs



def process_data():
    download_data()
    print('Preparing data to be model-ready ...')
    build_vocab('train.csv.zip')
    token2id('train.csv.zip')


def _pad_input(input_, size):
    return input_ + [config.PAD_ID] * (size - len(input_))


def get_batch(data_bucket, bucket_id, batch_size=1):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket
    encoder_size = config.BUCKETS[bucket_id]
    encoder_inputs = []

    for _ in range(batch_size):
        encoder_input = random.choice(data_bucket)
        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(_pad_input(encoder_input, encoder_size)))

    # now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    return batch_encoder_inputs, batch_decoder_inputs

if __name__ == '__main__':
    process_data()
