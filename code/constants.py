import os


# Root dir of data, expected to contain files train-*.tfrecord, used in inputpipeline.py
RECORDS_ROOT = os.environ.get('RECORDS_ROOT', 'data')

OTHER_CODECS_ROOT = os.environ.get('OTHER_CODECS_ROOT', 'other_codecs')

VALIDATION_DATASETS_ROOT = os.environ.get('VAL_ROOT', '')

CONFIG_BASE_AE = os.environ.get('CONFIG_BASE_AE', 'ae_configs')
CONFIG_BASE_PC = os.environ.get('CONFIG_BASE_PC', 'pc_configs')

NUM_PREPROCESS_THREADS = int(os.environ.get('NUM_PREPROCESS_THREADS', 4))
NUM_CROPS_PER_IMG = int(os.environ.get('NUM_CROPS_PER_IMG', 1))
