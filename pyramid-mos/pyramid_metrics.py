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
from scipy.stats import spearmanr

from data import Data, collate_custom
from pyramid_train import EncoderDecoder


def MAE(true_scores, pred_scores):
    err = np.sum(torch.abs(pred_scores - true_scores))/len(true_scores)
    return err

def RMSE(true_scores, pred_scores):
    mse = ((true_scores - pred_scores)**2).mean()
    err = np.sqrt(mse)
    return err

def Pearson(true_scores, pred_scores):
    true_mean = np.mean(true_scores)
    pred_mean = np.mean(pred_scores)

    tmp1 = np.sum((true_scores-true_mean)*(pred_scores-pred_mean))
    tmp2 = np.sqrt(np.sum((true_scores-true_mean)**2)*np.sum((pred_scores-pred_mean)**2))
    corr = tmp1/tmp2
    return corr

def Spearman(true_scores, pred_scores):

    def _get_ranks(x):
        tmp = x.argsort()
        ranks = torch.zeros_like(tmp)
        ranks[tmp] = torch.arange(len(x))
        return ranks

    x_rank = _get_ranks(true_scores)
    y_rank = _get_ranks(pred_scores)
    
    n = true_scores.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)



def inference(csv_dir, work_dir):
    device = torch.device("cuda:1")
    model = EncoderDecoder(input_dim=257, hidden_dim=256)
    model.load_state_dict(torch.load(os.path.join(work_dir, "pyramid_best.pth")))
    model = model.to(device)


    dataset = Data(csv_dir, mode='test')
    loader = data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_custom)
    
    print("Starting inference...")
    print("Test size: ", len(loader)*32)

    true_scores = []
    pred_scores = []

    for batch in tqdm(loader):
        aud = batch['aud'].to(device)
        lens = batch['lens']
        y_i = batch['score'].to(device).unsqueeze(-1).float()
        pred_y_i = model(aud, lens).float()

        true_scores.append(y_i.detach().cpu().numpy())
        pred_scores.append(pred_y_i.detach().cpu().numpy())

    true_scores = np.array(true_scores)
    pred_scores = np.array(pred_scores)

    print("Calculating metrics...")

    mean_abs_err = MAE(true_scores, pred_scores)
    r_mean_sq_err = RMSE(true_scores, pred_scores)
    PCC = Pearson(true_scores, pred_scores)
    SCC, p = spearmanr(true_scores, pred_scores)

    print("MAE: {:>3f}\nrMSE: {:>3f}\nPCC: {:>3f}\nSCC: {:>3f}".format(mean_abs_err, r_mean_sq_err, PCC, SCC))







inference("/nobackup/anakuzne/data/COSINE-orig/csv/all.csv", "/nobackup/anakuzne/models/mos-predict/")