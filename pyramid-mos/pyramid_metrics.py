import torch
import torch.utils.data as data
import torch.nn as nn

import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import soundfile as sfl
import librosa
from tqdm import tqdm
import copy
import sys

from data import Data, collate_custom
from pyramid_train import EncoderDecoder


def MAE():
    pass

def RMSE():
    pass

def Pearson():
    pass

def Spearman():
    pass



def inference(csv_dir, work_dir):
    device = torch.device("cuda:1")
    model = EncoderDecoder(input_dim=257, hidden_dim=256)
    model.load_state_dict(torch.load(os.path.join(work_dir, "pyramid_best.pth")))
    model = model.to(device)
   

    MSE = nn.MSELoss()
    MSE.to(device)


    dataset = Data(csv_dir, mode='test')
    loader = data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_custom)
    
    print("Starting inference...")
    print("Test size: ", len(loader)*32)

    for batch in tqdm(loader):
        aud = batch['aud'].to(device)
        lens = batch['lens']
        scores = batch['score'].to(device).unsqueeze(-1).float()
        pred_scores = model(aud, lens).float()
    


if __name__ == "__main__":
    inference("/nobackup/anakuzne/data/COSINE-orig/csv/all.csv", "/nobackup/anakuzne/models/mos-predict/")