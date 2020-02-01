import argparse
import os
from architectures.encoder_decoder_attentional_lstm import EncoderDecoderLSTM

parser = argparse.ArgumentParser()
parser.add_argument("--num", required=True, type=int, help="Number of questions to train and test with")
parser.add_argument("--id", default='0', type=str, help="Cude visible device id for use")

args = parser.parse_args()
number_questions = args.num
cuda_id = args.id

os.environ['CUDA_VISIBLE_DEVICES'] = cuda_id

network = EncoderDecoderLSTM(number_questions)
network.train()