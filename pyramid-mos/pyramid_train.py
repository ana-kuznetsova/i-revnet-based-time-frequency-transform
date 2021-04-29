import torch
import torch.utils.data as data
import torch.nn as nn


import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import soundfile as sfl
import librosa

from data import Data, collate_custom
from encoder import Encoder
from attn import Attention


class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.attn_decoder = Attention()

    def forward(self, x, lens):
        K, V, Q, lens = self.encoder(x, lens)
        attn_out = self.attn_decoder(Q, K, V)
        return attn_out



######### TRAINING LOOP ###########

csv_path = '/nobackup/anakuzne/data/COSINE-orig/csv/all.csv'
epochs = 100

device = torch.device("cuda:1")
model = EncoderDecoder(input_dim=257, hidden_dim=256)
model.to(device)

criterion = nn.MSELoss()
criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

dataset = Data(csv_path, mode='train')
loader = data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_custom)

for ep in range(1, epochs+1):
    for batch in loader:
        aud = batch['aud'].to(device)
        lens = batch['lens']
        scores = batch['score'].to(device)
        pred_scores = model(aud, lens)
        batch_loss = criterion(pred_scores, scores)
        print(batch_loss)
        