import torch
import torch.nn as nn
from torchtext.data import BucketIterator

import pandas as pd
import os

from data import create_data_iterators
from torchtext.data import Field, BucketIterator, Iterator, TabularDataset, Dataset
from definitions import ROOT_DIR


# TESTING PARAMETERS
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
q_type = 'algebra__linear_1d.tsv'
char_offset = True

"""
GRAB DATA
"""
_, _, SRC, TRG, _ = create_data_iterators(n_train=-1, q_type=q_type, device=device, difficulty='train-easy', batch_size='1')  # grab the SRC and TRG lexicons

data = TabularDataset(path=os.path.join(ROOT_DIR, 'data_test', q_type),
                              format='TSV',
                              fields=[('index', None), ('question', SRC), ('answer', TRG), ('source', None)],
                              skip_header=True)

test_iterator = Iterator(data,
                        batch_size=1,
                        sort=False,
                        train=False,
                        device=device)

"""
LOAD MODEL
"""
model = torch.load(os.path.join(ROOT_DIR, 'saved_models', f'FULLSET_train-easy_{q_type[:-4]}_ENCODER_DECODER_ATTENTIONAL_GRU.pt'))


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

df, score = test(model=model, iterator=test_iterator, input_itos=SRC.vocab.itos, output_itos=TRG.vocab.itos, output_stoi=TRG.vocab.stoi, char_offset=char_offset)
print(df)
print('')