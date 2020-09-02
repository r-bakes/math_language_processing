import argparse
import os

from parameters import q_list
from architectures.random_forest import random_forest_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--n", default=200000, type=int, help="Number of questions to train and test with")
parser.add_argument("--e", default=150, type=int, help="Number of epochs (only applicable to neural nets)")
parser.add_argument("--m", default='lstm', choices=['test', 'lstm', 'fst_clf'], type=str, help="model to train")
parser.add_argument("--v", default='onehot_char', choices=['tfid_char', 'tfid_word', 'onehot_char', 'onehot_word'], type=str, help="vectorization scheme to use (only applicable to random forest option)")
parser.add_argument("--q", default=q_list[0], choices=q_list, type=str, help="question type to train on")
parser.add_argument("--id", default='1', choices=['0','1','2','3'], type=str, help="Cuda visible device id for use")
parser.add_argument("--exp", default=f"BENCHMARK", type=str, help="Experiment name for file storing results" )
args = parser.parse_args()

n_questions = args.n
n_epochs = args.e
q_type = args.q
v_scheme = args.v
model = args.m
exp_name = args.exp
os.environ['CUDA_VISIBLE_DEVICES'] = args.id

if model == 'fst_clf':
    random_forest_experiment(n_train=n_questions, q_type=q_type)