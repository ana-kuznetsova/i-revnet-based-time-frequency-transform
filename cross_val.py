import torch
import torch.utils.data as data
import torch.nn as nn


import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import librosa
import random


class Data(data.Dataset):
    def __init__(self, clean_path, noisy_path, mode='train'):
        self.cnames = [os.path.join(clean_path, n) for n in os.listdir(clean_path)]
        self.nnames = [os.path.join(noisy_path, n) for n in os.listdir(noisy_path)]
        self.df = random.sample([(i, j) for i, j in zip(self.cnames, self.nnames)], len(self.cnames))
        self.train = self.df[:int(len(self.df)*0.7)]
        self.dev = self.df[int(len(self.df)*0.7):int(len(self.df)*0.8)]
        self.test = self.df[int(len(self.df)*0.8):]
        self.mode = mode
    
    def __len__(self):
        if self.mode=='train':
            return len(self.train)
        if self.mode=='dev':
            return len(self.dev)
        if self.mode=='test':
            return len(self.test)

    def read_aud(self, clean_path, noisy_path):
        clean, _ = librosa.core.load(clean_path, sr=16000)
        noisy, _ = librosa.core.load(noisy_path, sr=16000)
        res = np.concatenate([clean, noisy], axis=1)
        return torch.tensor(res)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.mode=='train':
            clean = self.train[idx][0]
            noisy = self.train[idx][1]
            res = self.read_aud(clean, noisy)
            return res
        if self.mode=='dev':
            clean = self.dev[idx][0]
            noisy = self.dev[idx][1]
            res = self.read_aud(clean, noisy)
            return res
        if self.mode=='test':
            clean = self.test[idx][0]
            noisy = self.test[idx][1]
            res = self.read_aud(clean, noisy)
            return res


dataset = Data(clean_path='/nobackup/anakuzne/data/COSINE-orig/clean-train',
               noisy_path='/nobackup/anakuzne/data/COSINE-orig/noisy-train', mode='train')

print(dataset)