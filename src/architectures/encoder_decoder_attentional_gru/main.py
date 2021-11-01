"""
Exeuction
"""
import torch

import torch.nn as nn
import torch.optim as optim

import time
import os

from src.data_preprocessing import create_data_iterators, epoch_time
from definitions import RESULTS_DIR, ROOT_DIR
from src import parameters as p


def encoder_decoder_attentional_gru_experiment(
        n_train: int,
        q_type: str,
        n_epochs: int,
        exp_name: str,
        difficulty: str,
        device_id: int,
        batch_size: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        char_offset: bool,
        save_model: bool) -> None:

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