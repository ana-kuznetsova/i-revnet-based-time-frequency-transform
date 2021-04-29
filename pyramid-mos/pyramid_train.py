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




######### TRAINING LOOP ###########

csv_path = '/nobackup/anakuzne/data/COSINE-orig/csv/all.csv'
epochs = 100

device = torch.device("cuda:1")
model = Encoder(input_dim=257, hidden_dim=256)
model.to(device)
attention = Attention(input_dim=128, out_dim=1)
attention.to(device)

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
        keys, values, queries, lens = model(aud, lens)
        attn_out = attention(queries, keys, values)
        