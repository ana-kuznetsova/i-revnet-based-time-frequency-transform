from librosa.core import audio
import numpy as np
from torch.utils import data
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pBLSTM_encoder import Encoder
from pLSTM_decoder import Decoder
from data import Data, collate_custom
import torch.utils.data as data


class EncodeDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, lens, device):
        keys, values, lens = self.encoder(source, lens)
        #torch.Size([59, 5, 128])  TxBx*
        return self.decoder(keys, values, lens, device, source)




'''
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
'''
device = torch.device('cuda:1')
embedding_dim = 1
input_dim = 513
hidden_dim = 128
dropout = 0.5
learning_rate = 0.0001


encoder = Encoder(input_dim, hidden_dim)
decoder = Decoder(embedding_dim, hidden_dim)

model = EncodeDecoder(encoder, decoder)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()


csv_path = '/nobackup/anakuzne/data/COSINE-orig/csv/all.csv'

dataset = Data(csv_path, mode='train')
loader = data.DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=collate_custom)

dataset_dev = Data(csv_path, mode='dev')
loader_dev = data.DataLoader(dataset_dev, batch_size=5, shuffle=False, collate_fn=collate_custom)

for batch in loader:
    audio = batch['aud'].to(device)
    lens = batch['lens']
    mos = batch['score'].to(device)
    pred = model(audio, lens, device)
    print("Model out:", pred)