import os
import datetime
# dir_results = os.path.join(ROOT_DIR, "results", "encoder_decoder_lstm_001.txt")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATA_TSV_DIR = os.path.join(ROOT_DIR, 'data_tsv')

RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
LOGDIR = os.path.join(ROOT_DIR, "logs","scalars")
