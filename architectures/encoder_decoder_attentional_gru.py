import torch
import torchtext

import math
import pandas as pd
import numpy as np

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
from definitions import RESULTS_DIR, ROOT_DIR
import parameters as p



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

    def forward(self, src: Tensor, trg: Tensor=None, teacher_forcing_ratio: float=0.5, output_stoi=None, char_offset=False, offset=4) -> Tensor:

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

        elif trg is None and char_offset is False:  # Prediction sampling
            outputs = torch.zeros(p.max_answer_length, batch_size, trg_vocab_size).to(self.device)

            # first input to the decoder is the <sos> token
            output = torch.full(size=(1, batch_size), fill_value=output_stoi['<SOS>'], dtype=int).to(self.device)[0]

            i = 1
            while i < p.max_answer_length and output[0] != output_stoi['<EOS>']:
                output, hidden = self.decoder(output, hidden, encoder_outputs)
                outputs[i] = output
                output = output.max(1)[1]
                i+=1

            outputs = outputs[0:i]  # trim before returning

        elif trg is None and char_offset is True:  # Prediction sampling
            outputs = [torch.zeros(p.max_answer_length, batch_size, trg_vocab_size).to(self.device) for i in range(0, offset)]

            # first input to the decoder is the <sos> token, generate first output char and select top k
            output_init = torch.full(size=(1, batch_size), fill_value=output_stoi['<SOS>'], dtype=int).to(self.device)[0]
            output_init, hidden_init = self.decoder(output_init, hidden, encoder_outputs)

            for outputs_i in range(0, offset):

                output = output_init.topk(offset)[1][0][outputs_i:outputs_i+1] # set up first char of sequence to be the i_th prediction offset
                hidden = hidden_init
                outputs[outputs_i][1][0][output[0]] = 1  # manually set first step of prediction tensor to have index of desired offset char prediction

                i = 2
                while i < p.max_answer_length and output[0] != output_stoi['<EOS>']:
                    output, hidden = self.decoder(output, hidden, encoder_outputs)
                    outputs[outputs_i][i] = output
                    output = output.max(1)[1]
                    i+=1

                outputs[outputs_i] = outputs[outputs_i][0:i]  # trim before returning

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
         char_offset: bool,
         iterator: BucketIterator,
         output_itos: list,
         input_itos: list,
         output_stoi: dict):

    model.eval()
    questions, solutions, predictions = [], [], []
    predictions_2, predictions_3, predictions_4 = [], [], []
    score=0
    with torch.no_grad():

        for _, batch in enumerate(iterator):

            src = batch.question
            trg = batch.answer

            output = model(src=src, trg=None, output_stoi=output_stoi, char_offset=char_offset)  # turn off teacher forcing


            question_sequence = src.T
            solution_sequence = trg[1:].T

            if char_offset == False:
                topv, topi = output[1:].topk(1)
                prediction_sequence = topi[:, :, 0].T

                for pred_seq, sol_seq, que_seq in zip(prediction_sequence, solution_sequence, question_sequence):
                    question = ''.join([input_itos[i] for i in que_seq])
                    prediction = ''.join([output_itos[i] for i in pred_seq])
                    solution = ''.join([output_itos[i] for i in sol_seq])

                    questions.append(question)
                    predictions.append(prediction)
                    solutions.append(solution)

                    if prediction == solution: score += 1

                df = pd.DataFrame(data={'questions': questions, 'solutions': solutions, 'predictions': predictions})

            else:
                topv, topi = output[0][1:].topk(1)
                prediction_sequence = topi[:, :, 0].T

                _, topi2 = output[1][1:].topk(1)
                prediction2_sequence = topi2[:, :, 0].T

                _, topi3 = output[2][1:].topk(1)
                prediction3_sequence = topi3[:, :, 0].T

                _, topi4 = output[3][1:].topk(1)
                prediction4_sequence = topi4[:, :, 0].T


                for pred_seq, sol_seq, que_seq, pred2, pred3, pred4 in zip(prediction_sequence, solution_sequence, question_sequence, prediction2_sequence, prediction3_sequence, prediction4_sequence):
                    question = ''.join([input_itos[i] for i in que_seq])
                    prediction = ''.join([output_itos[i] for i in pred_seq])
                    solution = ''.join([output_itos[i] for i in sol_seq])

                    prediction2 = ''.join([output_itos[i] for i in pred2])
                    prediction3 = ''.join([output_itos[i] for i in pred3])
                    prediction4 = ''.join([output_itos[i] for i in pred4])

                    questions.append(question)
                    predictions.append(prediction)
                    solutions.append(solution)

                    predictions_2.append(prediction2)
                    predictions_3.append(prediction3)
                    predictions_4.append(prediction4)

                    if prediction == solution: score += 1

                df = pd.DataFrame(data={'questions': questions, 'solutions': solutions, 'predictions': predictions, 'predictions2': predictions_2, 'predictions3': predictions_3,  'predictions4': predictions_4})

    print(f'FINAL SCORE: {score/len(questions)}')

    return df, score/len(questions)

def encoder_decoder_attentional_gru_experiment(n_train, q_type, n_epochs, exp_name, difficulty, device_id, batch_size, encoder_hidden_size, decoder_hidden_size, char_offset, save_model):
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start = time.time()

    # Grab Data
    train_iterator, valid_iterator, SRC, TRG, test_iterator = create_data_iterators(n_train=n_train, q_type=q_type, difficulty=difficulty, device=device, batch_size=batch_size)

    # Model Parameters
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    ENC_EMB_DIM = 512
    DEC_EMB_DIM = 512
    ENC_HID_DIM = encoder_hidden_size
    DEC_HID_DIM = decoder_hidden_size
    ATTN_DIM = 16
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    CLIP = 0.1

    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

    attentional = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attentional)

    model = Seq2Seq(encoder, decoder, device).to(device)

    best_valid_loss = float('inf')

    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), betas=(p.beta1, p.beta2), lr=p.learning_rate, eps=p.epsilon)

    PAD_IDX = TRG.vocab.stoi['<pad>']

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    for epoch in range(1, n_epochs):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s \t\t\t\t Question: {q_type[:-4]} | Experiment: {exp_name} | Est. Time Remaining: {round((n_epochs-epoch)*(end_time - start_time)/(60**2),2)}h')
        else:
            print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')

    results, score = test(model=model, iterator=test_iterator, input_itos=SRC.vocab.itos, output_itos=TRG.vocab.itos, output_stoi=TRG.vocab.stoi, char_offset=char_offset)

    if not os.path.exists(os.path.join(RESULTS_DIR, exp_name.lower())): os.makedirs(os.path.join(RESULTS_DIR, exp_name.lower()))

    results.to_csv(os.path.join(RESULTS_DIR, exp_name.lower(), f'{exp_name}_{difficulty}_{q_type[:-4]}_ENCODER_DECODER_ATTENTIONAL_GRU.tsv'), sep='\t', index=False)

    with open(os.path.join(RESULTS_DIR, exp_name.lower(), f'{exp_name}_{difficulty}_{q_type[:-4]}_ENCODER_DECODER_ATTENTIONAL_GRU.tsv'), 'r') as result_file:
        with open(os.path.join(RESULTS_DIR, exp_name.lower(), f'copy_{exp_name}_{difficulty}_{q_type[:-4]}.tsv'), 'w') as final_file:
            final_file.write(f'experiment: {exp_name} | q_type: {q_type} | score: {round(score, 4)} | model: ENCODER DECODER ATTENTIONAL GRU | n_train: {len(train_iterator.dataset)} | n_epochs: {n_epochs} | difficulty: {difficulty} | hours_training: {round((time.time() - start)/(60**2), 2)} | batch size: {batch_size} | optimizer: adam | criterion: cross entropy loss | enc hidden dim: {ENC_HID_DIM} | dec hidden dim: {DEC_HID_DIM} | attn dim: {ATTN_DIM}\n')
            final_file.write(result_file.read())
    os.rename(os.path.join(RESULTS_DIR, exp_name.lower(), f'copy_{exp_name}_{difficulty}_{q_type[:-4]}.tsv'), os.path.join(RESULTS_DIR, exp_name.lower(), f'{exp_name}_{difficulty}_{q_type[:-4]}_ENCODER_DECODER_ATTENTIONAL_GRU.tsv'))

    if save_model is True:
        torch.save(model, os.path.join(ROOT_DIR, 'saved_models', f'{exp_name}_{difficulty}_{q_type[:-4]}_ENCODER_DECODER_ATTENTIONAL_GRU.pt'))

    print(f'{q_type[:-4]} EXPERIMENT CONCLUDED IN {round((time.time() - start)/(60**2), 2)} HOURS')
