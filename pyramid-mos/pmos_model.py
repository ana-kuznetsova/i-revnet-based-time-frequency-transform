import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field, BucketIterator
import spacy
import numpy as np
import random
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


######
#psedo 
######
def create_inout_sequences():
	# read audio file
	# source = audio_tf 
	# train_label = mos_score

    inout_seq = []

    for i in range(len(source)):
        train_seq = audio_tf[i]
        train_label = mos_score[i]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


class EncodeDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.max_len = 200
        self.decoder = decoder

    def forward(self, source, target):
        encoder_outputs, hidden = self.encoder(source, self.max_len)
        return self.decoder(encoder_outputs)


def create_model(source, target):
    # Define the required dimensions and hyper parameters
    embedding_dim = 1
    input_dim = 200
    hidden_dim = 128
    dropout = 0.5
    learning_rate = 0.0001

    # Instantiate the models
	encoder = pLSTM_encoder(input_dim, hidden_dim)
	dencoder = pLSTM_decoder(embedding_dim,hidden_dim)

    model = EncodeDecoder(encoder, decoder)

    model = model.to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    return model, optimizer, criterion


train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
model, optimizer, criterion = create_model(source, target)

num_epochs = 2000

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()

        H = encoder(seq)
        y_pred = dencoder(H) #predict mos

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')


print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')