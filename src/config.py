from pathlib import Path
import os


ROOT = str(Path('__file__').absolute().parent)
DATA_PATH = os.path.join(ROOT, 'data')
PROCESSED_PATH = os.path.join(ROOT, 'data', 'processed')
CPT_PATH = os.path.join(ROOT, 'data', 'checkpoints')
GLOVE_PATH = os.path.join(DATA_PATH, 'glove.6B.zip')
VOCAB_PATH = os.path.join(PROCESSED_PATH, 'vocab.txt')
THRESHOLD = 10
PAD_ID = 0
BATCH_SIZE = 64
EMBEDDING_DIMENSION = 50
GLOVE_SIZE = 50
NUM_CLASSES = 6
HIDDEN_LAYER_SIZE = 128
MAX_SEQ_LENGTH = 100
KEEP_PROB = 0.8
ATTENTION_SIZE = 128
LR = 0.01
MAX_GRAD_NORM = 1.0
PRE_TRAINED = True
EPOCHS = 400
VOCAB_SIZE = 19258
