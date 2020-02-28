import argparse
import os
import parameters as p
from architectures.encoder_decoder_attentional_lstm import EncoderDecoderLSTM
from architectures.random_forest import RandomForestWord

from architectures.test import Test


parser = argparse.ArgumentParser()
parser.add_argument("--n", default=2, type=int, help="Number of questions to train and test with")
parser.add_argument("--e", default=10, type=int, help="Number of epochs (only applicable to neural nets)")
parser.add_argument("--m", default='lstm', choices=['test', 'lstm', 'rnd_fst'], type=str, help="model to train")
parser.add_argument("--v", default='onehot_char', choices=['tfid_char', 'tfid_word', 'onehot_char', 'onehot_word'], type=str, help="vectorization scheme to use (only applicable to random forest option)")
parser.add_argument("--q", default=p.q_list[0], choices=p.q_list, type=str, help="question type to train on")
parser.add_argument("--id", default='1', choices=['0','1','2','3'], type=str, help="Cuda visible device id for use")
args = parser.parse_args()

n_questions = args.n
n_epochs = args.e
q_type = args.q
v_scheme = args.v
model = args.m
os.environ['CUDA_VISIBLE_DEVICES'] = args.id

if model == 'test':
    network = Test(n_questions,
                   n_epochs,
                   q_type)
elif model == 'rnd_fst':
    network = RandomForestWord(n_train=n_questions,
                               q_type=q_type,
                               analysis=v_scheme)
elif model == 'lstm':
    network = EncoderDecoderLSTM(n_train=n_questions,
                                 n_epochs=n_epochs,
                                 q_type=q_type)

network.train()