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
        self.df = pd.read_csv(csv_path).sample(frac=1, random_state=42).reset_index(drop=True)
        self.train = self.df[:int(len(self.df)*0.7)]
        print(len(self.train))
        self.dev = self.df[int(len(self.df)*0.7):int(len(self.df)*0.8)]
        print(len(self.dev))
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


def replaceZeroes(data):
  min_nonzero = np.min(data[np.nonzero(data)])
  data[data == 0] = min_nonzero
  return data

def collate_custom(batch_data, maxlen=751):
    batch_aud = []
    batch_scores = []
    lens = []
    tmp = []

    for ex in batch_data:
        aud, score = ex
        aud, _ = librosa.core.load(aud, sr=16000)
        aud = replaceZeroes(librosa.stft(aud, n_fft=512))
        aud = np.abs(10*np.log10(aud))
        tmp.append((aud, score, aud.shape[1]))

    tmp = sorted(tmp, key=lambda x: x[-1], reverse=True)
    for i in tmp:
        aud = i[0]
        aud = nn.ZeroPad2d(padding=(0, maxlen-aud.shape[1], 0, 0))(torch.tensor(aud))
        batch_aud.append(aud)
        batch_scores.append(torch.tensor(i[1]))
        lens.append(torch.tensor(i[-1]))
    return {"aud":torch.stack(batch_aud), "lens": torch.stack(lens), "score":torch.stack(batch_scores)}

if __name__ == "__main__":
    csv_path = '/nobackup/anakuzne/data/COSINE-orig/csv/all.csv'

    dataset = Data(csv_path, mode='train')
    loader = data.DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=collate_custom)

    dataset_dev = Data(csv_path, mode='dev')
    loader_dev = data.DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=collate_custom)
