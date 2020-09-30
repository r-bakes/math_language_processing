import argparse
import pandas as pd
import numpy as np
import os

from parameters import q_list
from definitions import RESULTS_DIR
from architectures.random_forest import random_forest_experiment
from architectures.encoder_decoder_attentional_gru import encoder_decoder_attentional_gru_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--n", default=-1, type=int, help="Number of questions to train and test with")
parser.add_argument("--e", default=150, type=int, help="Number of epochs (only applicable to neural nets)")
parser.add_argument("--d", default='train-easy', type=str, choices=['train-easy', 'train-medium', 'train-hard', 'train-all'], help='difficulty of training dataset')
parser.add_argument("--m", default='gru', choices=['gru', 'lstm', 'fst_clf'], type=str, help="model to train")
parser.add_argument("--eh", default=2048, choices=[64, 128, 256, 512, 1024, 2048], type=int, help="encoder hidden size")
parser.add_argument("--dh", default=2048, choices=[64, 128, 256, 512, 1024, 2048], type=int, help="decoder hidden size")

parser.add_argument("--q", default=q_list[0], choices=q_list, type=str, help="question type to train on")
parser.add_argument("--exp", default=f"BENCHMARK", type=str, help="Experiment name for file storing results" )
parser.add_argument("--bs", default=512, choices=[16, 32, 64, 128, 256, 512, 1024], type=int, help="batch size for training")
parser.add_argument("--co", default=True, choices=[True, False], type=bool, help="char offsetting top 4 on/off")


parser.add_argument("--v", default='char', choices=['char', 'word'], type=str, help="vectorization scheme to use (only applicable to random forest option)")
parser.add_argument("--id", default='1', choices=['0','1','2','3'], type=str, help="Cuda visible device id for use")
parser.add_argument("--md", default=6, type=int, help='max depth for random forest')
parser.add_argument("--s", default=10, type=int, help='max number of estimators for random forest')

args = parser.parse_args()

n_questions = args.n
n_epochs = args.e
model = args.m
enc_hidden = args.eh
dec_hidden = args.dh
difficulty = args.d
q_type = args.q
char_offset = args.co

v_scheme = args.v
exp_name = args.exp
batch_size = args.bs

max_depth = args.d
n_estimators = args.s
id = args.id

if model == 'fst_clf':

    for q in q_list:
        try:

            result = random_forest_experiment(n_train=n_questions, q_type=q, analyzer=v_scheme, max_depth=max_depth, n_estimators=n_estimators)
            result = pd.DataFrame(result)

        except MemoryError:
            result = pd.DataFrame({'question': [q], 'score': [np.nan]})

        try:
            results = pd.read_csv(os.path.join(RESULTS_DIR, f'{exp_name}_random_forest_results__{n_questions}_n_train_{n_estimators}_estimators_{max_depth}_maxdepth.txt'))
            results = results.append(result)

            results.to_csv(os.path.join(RESULTS_DIR, f'{exp_name}_random_forest_results__{n_questions}_n_train_{n_estimators}_estimators_{max_depth}_maxdepth.txt'), index=False)
        except FileNotFoundError:
            result.to_csv(os.path.join(RESULTS_DIR, f'{exp_name}_random_forest_results__{n_questions}_n_train_{n_estimators}_estimators_{max_depth}_maxdepth.txt'), index=False)

elif model=='gru':
    encoder_decoder_attentional_gru_experiment(n_train=n_questions,
                                               q_type=q_type,
                                               n_epochs=n_epochs,
                                               exp_name=exp_name,
                                               difficulty=difficulty,
                                               device_id=id,
                                               batch_size=batch_size,
                                               encoder_hidden_size=enc_hidden,
                                               decoder_hidden_size=dec_hidden,
                                               char_offset=char_offset)

# elif model=='lstm':
#     encoder_decoder_attentional_lstm_experiment(n_train=n_questions,
#                                                 q_type=q_type,
#                                                 n_epochs=n_epochs,
#                                                 exp_name=exp_name,
#                                                 difficulty=difficulty,
#                                                 device_id=id,
#                                                 batch_size=batch_size,
#                                                 encoder_hidden_size=enc_hidden,
#                                                 decoder_hidden_size=dec_hidden)