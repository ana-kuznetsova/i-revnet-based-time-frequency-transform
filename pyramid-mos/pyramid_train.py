import torchaudio
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




######### TRAINING LOOP ###########

csv_path = '/nobackup/anakuzne/data/COSINE-orig/csv/all.csv'
epochs = 100

device = torch.device("cuda:1")
model = Encoder(input_dim=257, hidden_dim=256)
model.to(device)

criterion = nn.MSELoss()
criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

dataset = Data(csv_path, mode='train')
loader = data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_custom)

for ep in range(1, epochs+1):
    for batch in loader:
        aud, lens, scores = batch.to(device)
        keys, values, lens = model(aud, lens)
        print(keys)