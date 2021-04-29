import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as weight_init


import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import soundfile as sfl
import librosa
import wandb
import copy

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
#wandb.init(project='i-rev-net-verify', entity='anakuzne')



csv_path = '/nobackup/anakuzne/data/COSINE-orig/csv/all.csv'
work_dir = '/nobackup/anakuzne/models/mos-predict/'
epochs = 100
batch_size = 32

device = torch.device("cuda:1")
model = EncoderDecoder(input_dim=257, hidden_dim=256)
#weight_init.xavier_normal(model)
model.to(device)

criterion = nn.MSELoss()
criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

dataset = Data(csv_path, mode='train')
loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_custom)

dataset_dev = Data(csv_path, mode='dev')
loader_dev = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_custom)


best = copy.deepcopy(model.state_dict())
prev_val=99999

#wandb.watch(model)

for ep in range(1, epochs+1):
    epoch_loss = 0

    for batch in loader:
        aud = batch['aud'].to(device)
        lens = batch['lens']
        scores = batch['score'].to(device).unsqueeze(-1)
        pred_scores = model(aud, lens)
        batch_loss = criterion(pred_scores, scores).float()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        batch_loss = batch_loss.detach().cpu().numpy()
        epoch_loss+=batch_loss

    print('Epoch:{:2} Training loss:{:>4f}'.format(epoch, float(epoch_loss/len(loader))))
    #wandb.log({"train_loss": epoch_loss/len(loader)})

    if epoch%5==0:
        ##Validation
        overall_val_loss = 0
        for batch in loader_dev:
            aud = batch['aud'].to(device)
            lens = batch['lens']
            scores = batch['score'].to(device).unsqueeze(-1)
            pred_scores = model(aud, lens)
            batch_loss = criterion(pred_scores, scores)

            overall_val_loss+=batch_loss.detach().cpu().numpy()
            curr_val_loss = overall_val_loss/len(loader_dev)
        
        if curr_val_loss < prev_val:
            torch.save(best, os.path.join(work_dir, 'pyramid_best.pth'))
            prev_val = curr_val_loss
        
        print('Validation loss: ', curr_val_loss)
        #wandb.log({"val_loss": curr_val_loss})

        torch.save(best, os.path.join(work_dir, "pyramid_last.pth"))

        