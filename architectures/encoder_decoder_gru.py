from definitions import DATA_TSV_DIR, DATA_DIR

from io import open
import numpy as np
import unicodedata
import string
import re
import random
import os

import math

import torch
import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from parameters import max_question_length


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
teacher_forcing_ratio = 0.5
MAX_LENGTH = max_question_length
SOS_token = 0
EOS_token = 1
PAD_token = 2


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS', 2: 'PAD'}
        self.n_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word]+1

def read_langs(n_train, q_type):
    print('reading data...')
    train_data = get_data(n_train=n_train, q_type=q_type, type='train')
    test_data = get_data(n_train=n_train, q_type=q_type, type='test')

    test_data = np.char.lower(test_data)
    train_data = np.char.lower(train_data)

    input_lang = Lang('questions')
    output_lang = Lang('answers')

    return input_lang, output_lang, train_data, test_data

def prepare_data(n_train, q_type):
    input_lang, output_lang, train_data, test_data = read_langs(n_train, q_type)

    print('Counting words...')
    for i, o in train_data:
        input_lang.add_sentence(i)
        output_lang.add_sentence(o)
    for i, o in test_data:
        input_lang.add_sentence(i)
        output_lang.add_sentence(o)
    print('Counted words:')
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, train_data, test_data



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, input, hidden):
#         output = self.embedding(input).view(1, 1, -1)
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#         output = self.softmax(self.out(output[0]))
#         return output, hidden
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return input_tensor, target_tensor


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train_iters(encoder, decoder, epochs, data, input_lang, output_lang, learning_rate=0.01):
    start = time.time()


    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    n_epoch=1
    while n_epoch <= epochs:
        print_loss_total = 0  # Reset every print_every
        np.random.shuffle(data)
        for pair in data:
            input_tensor, target_tensor = tensorsFromPair(pair, input_lang, output_lang)


            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss


        print_loss_avg = print_loss_total / len(data)
        print_loss_total = 0
        print('%s (%d %d%%) %.4f' % (time_since(start, n_epoch / epochs),
                                     n_epoch, n_epoch / epochs * 100, print_loss_avg))
        n_epoch+=1



def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluate_test_data(encoder, decoder, data, input_lang, output_lang):
    score = 0
    for question, answer in data:
        print('>', question)
        print('=', answer)
        output_words, attentions = evaluate(encoder, decoder, question, input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

        if output_sentence.replace(' <EOS>', '') == answer: score+=1

    print('---------------------------------------------')
    print(f'SCORE: {score/len(data)}')


def encoder_decoder_gru_experiment(n_train, q_type, epochs):
    # BATCH_SIZE = 2
    # N_EPOCHS = 100
    # CLIP = 1
    #
    # SRC = Field(tokenize=list,
    #             lower=True)
    #
    # TRG = Field(tokenize=list,
    #             init_token='<SOS>',
    #             eos_token='<EOS>',
    #             lower=True)
    #
    # data = TabularDataset(path=os.path.join(DATA_TSV_DIR, 'train-easy', q_type),
    #                       format='TSV',
    #                       fields=[('index', None), ('question', SRC), ('answer', TRG)],
    #                       skip_header=True)
    #
    # data.examples = data.examples[0:n_train]  # reduce scope for testing

    input_lang, output_lang, train_data, test_data = prepare_data(n_train=n_train, q_type=q_type)  # Encoder reads an input sequence and outputs a single vector, decoder reads that vector and outputs a sequence

    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    # encoder1 = EncoderRNN(len(SRC.vocab), hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    # attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    print('Begin training\n')
    train_iters(encoder1, attn_decoder1, epochs=epochs, data=train_data, input_lang=input_lang, output_lang=output_lang)

    evaluate_test_data(encoder1, attn_decoder1, test_data, input_lang, output_lang)