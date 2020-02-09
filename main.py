import argparse
import os
from architectures.encoder_decoder_attentional_lstm import EncoderDecoderLSTM
from architectures.test import Test

parser = argparse.ArgumentParser()
parser.add_argument("--n", default=300, type=int, help="Number of questions to train and test with")
parser.add_argument("--e", default=2, type=int, help="Number of epochs")
parser.add_argument("--m", default='test', choices=['test','lstm'], type=str, help="model to train")
parser.add_argument("--id", default='0', choices=['0','1','2','3'], type=str, help="Cuda visible device id for use")
args = parser.parse_args()

n_questions = args.n
n_epochs = args.e
model = args.m

os.environ['CUDA_VISIBLE_DEVICES'] = args.id
# network = EncoderDecoderLSTM(n_questions, n_epochs)
if model == 'test':
    network = Test(n_questions, n_epochs)
elif model == 'lstm':
    network = EncoderDecoderLSTM(n_questions, n_epochs)
network.train()