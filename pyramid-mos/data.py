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



class Data(data.Dataset):
    def __init__(self, csv_path, mode='train'):
        self.df = pd.read_csv(csv_path).sample(frac=1, random_state=42)
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

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.mode=='train':
            return (self.train['path'][idx], self.train['MOS'][idx])
        if self.mode=='dev':
            return (self.dev['path'][idx], self.dev['MOS'][idx])
        if self.mode=='test':
            return (self.test['path'][idx], self.test['MOS'][idx])