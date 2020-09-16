import torch
import torchtext

import math
import pandas as pd

import random
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from torchtext.data import BucketIterator
import time
import os

from data import create_data_iterators, epoch_time
from definitions import RESULTS_DIR
import parameters as p

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor) -> Tuple[Tensor]:

        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:

        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim = 2)))

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:

        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep


    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim = 1))

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: Tensor, trg: Tensor=None, teacher_forcing_ratio: float=0.5, output_stoi=None) -> Tensor:

        batch_size = src.shape[1]
        trg_vocab_size = self.decoder.output_dim

        encoder_outputs, hidden = self.encoder(src)
        
        if trg is not None:  # Training
            outputs = torch.zeros(trg.shape[0], batch_size, trg_vocab_size).to(self.device)
            # first input to the decoder is the <sos> token
            output = trg[0, :]

            for t in range(1, trg.shape[0]):
                output, hidden = self.decoder(output, hidden, encoder_outputs)
                outputs[t] = output
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = output.max(1)[1]
                output = (trg[t] if teacher_force else top1)

        elif trg is None:  # Prediction sampling
            outputs = torch.zeros(p.max_answer_length, batch_size, trg_vocab_size).to(self.device)

            # first input to the decoder is the <sos> token
            output = torch.full(size=(1, batch_size), fill_value=output_stoi['<SOS>'], dtype=int)[0]

            i = 1
            while i < p.max_answer_length and output[0] != output_stoi['<EOS>']:
                output, hidden = self.decoder(output, hidden, encoder_outputs)
                outputs[i] = output
                output = output.max(1)[1]
                i+=1

            outputs = outputs[0:i]  # trim before returning

        return outputs

def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model: nn.Module,
          iterator: BucketIterator,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0
    for _, batch in enumerate(iterator):

        src = batch.question
        trg = batch.answer

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model: nn.Module,
             iterator: BucketIterator,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, batch in enumerate(iterator):

            src = batch.question
            trg = batch.answer

            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def test(model: nn.Module,
         iterator: BucketIterator,
         output_itos: list,
         input_itos: list,
         output_stoi: dict):

    model.eval()
    questions, solutions, predictions=[], [], []
    score=0
    with torch.no_grad():

        for _, batch in enumerate(iterator):

            src = batch.question
            trg = batch.answer

            output = model(src=src, trg=None, output_stoi=output_stoi)  # turn off teacher forcing

            topv, topi = output[1:].topk(1)
            question_sequence = src.T
            prediction_sequence = topi[:,:,0].T
            solution_sequence = trg[1:].T


            for pred_seq, sol_seq, que_seq in zip(prediction_sequence, solution_sequence, question_sequence):
                question = ''.join([input_itos[i] for i in que_seq])
                prediction = ''.join([output_itos[i] for i in pred_seq])
                solution = ''.join([output_itos[i] for i in sol_seq])

                questions.append(question)
                predictions.append(prediction)
                solutions.append(solution)
                if prediction == solution: score += 1

    print(f'FINAL SCORE: {score/len(questions)}')
    df = pd.DataFrame(data={'questions': questions, 'solutions': solutions, 'predictions': predictions})

    return df

def encoder_decoder_attentional_gru_experiment(n_train, q_type, n_epochs, exp_name):
    start = time.time()

    # Grab Data
    train_iterator, valid_iterator, SRC, TRG, test_iterator = create_data_iterators(n_train=n_train, q_type=q_type, device=device)

    # Model Parameters
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    ENC_EMB_DIM = 512
    DEC_EMB_DIM = 512
    ENC_HID_DIM = 2048
    DEC_HID_DIM = 2048
    ATTN_DIM = 1024
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    CLIP = 0.1

    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

    attentional = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attentional)

    model = Seq2Seq(encoder, decoder, device).to(device)

    best_valid_loss = float('inf')

    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters())

    PAD_IDX = TRG.vocab.stoi['<pad>']

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    for epoch in range(n_epochs):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')

    results = test(model=model, iterator=test_iterator, input_itos=SRC.vocab.itos, output_itos=TRG.vocab.itos, output_stoi=TRG.vocab.stoi)
    results.to_csv(os.path.join(RESULTS_DIR, f'{exp_name}_{q_type[:-4]}_ENCODER_DECODER_ATTENTIONAL_GRU.tsv'), sep='\t')

    print(f'EXPERIMENT CONCLUDED IN {(time.time() - start)/(60**2)} HOURS')

