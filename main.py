import argparse
import os
from architectures.encoder_decoder_attentional_lstm import EncoderDecoderLSTM

parser = argparse.ArgumentParser()
parser.add_argument("--n", required=True, type=int, help="Number of questions to train and test with")
parser.add_argument("--e", default=100, type=int, help="Number of epochs")
parser.add_argument("--id", default='0', type=str, help="Cuda visible device id for use")


args = parser.parse_args()
n_questions = args.n
n_epochs = args.e
cuda_id = args.id

os.environ['CUDA_VISIBLE_DEVICES'] = cuda_id

network = EncoderDecoderLSTM(n_questions, n_epochs)
network.train()