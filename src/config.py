from pathlib import Path
import os


ROOT = str(Path('__file__').absolute().parent)
DATA_PATH = os.path.join(ROOT, 'data')
LOG_DIR = os.path.join(DATA_PATH, 'log')
PROCESSED_PATH = os.path.join(ROOT, 'data', 'processed')
CPT_PATH = os.path.join(ROOT, 'data', 'checkpoints')
GLOVE_PATH = os.path.join(DATA_PATH, 'glove.6B.zip')
VOCAB_PATH = os.path.join(PROCESSED_PATH, 'vocab.txt')
THRESHOLD = 10
TRAIN_TEST_RATIO = 0.2
PAD_ID = 0
BATCH_SIZE = 128
EMBEDDING_DIMENSION = 100
GLOVE_SIZE = 100
NUM_CLASSES = 6
HIDDEN_LAYER_SIZE = 128
MAX_SEQ_LENGTH = 250
KEEP_PROB = 0.6
ATTENTION_SIZE = 256
LR = 0.001
MAX_GRAD_NORM = 1.0
PRE_TRAINED = False
TRAINABLE_EMBEDDING = True
EPOCHS = 10000
RETURN_ALPHA = False
VOCAB_SIZE = 42047