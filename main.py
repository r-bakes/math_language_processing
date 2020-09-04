import argparse
import pandas as pd
import numpy as np
import os

from parameters import q_list
from definitions import RESULTS_DIR
from architectures.random_forest import random_forest_experiment
from architectures.encoder_decoder_gru import encoder_decoder_gru_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--n", default=200000, type=int, help="Number of questions to train and test with")
parser.add_argument("--e", default=50, type=int, help="Number of epochs (only applicable to neural nets)")
parser.add_argument("--m", default='gru', choices=['gru', 'lstm', 'fst_clf'], type=str, help="model to train")
parser.add_argument("--v", default='char', choices=['char', 'word'], type=str, help="vectorization scheme to use (only applicable to random forest option)")
parser.add_argument("--q", default=q_list[0], choices=q_list, type=str, help="question type to train on")
parser.add_argument("--id", default='1', choices=['0','1','2','3'], type=str, help="Cuda visible device id for use")
parser.add_argument("--d", default=6, type=int, help='max depth for random forest')
parser.add_argument("--s", default=10, type=int, help='max number of estimators for random forest')
parser.add_argument("--exp", default=f"BENCHMARK", type=str, help="Experiment name for file storing results" )
args = parser.parse_args()

n_questions = args.n
n_epochs = args.e
q_type = args.q
v_scheme = args.v
model = args.m
exp_name = args.exp
max_depth = args.d
n_estimators = args.s
os.environ['CUDA_VISIBLE_DEVICES'] = args.id 

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

elif model == 'gru':
    encoder_decoder_gru_experiment(n_train=n_questions, q_type=q_type, epochs=n_epochs)