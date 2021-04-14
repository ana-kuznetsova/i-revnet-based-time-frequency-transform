import torchaudio
import torch
import torch.utils.data as data
import torch.nn as nn


import os
import numpy as np
import librosa
from pydub import AudioSegment
import pandas as pd


def segment(aud, ms=40, overlap=10):
    segments = []
    
    prev_count = 0
    count=0
    while count < len(aud):
        count+=ms
        aud_seg = aud[prev_count:count]
        aud_seg = np.asarray(aud_seg.get_array_of_samples(), dtype = np.int32)
        max_amplitude = max(aud_seg)
        aud_seg = aud_seg/max_amplitude
        segments.append(aud_seg)
        prev_count=count-overlap
    return segments

def collate_custom(data):
    '''
    For batch
    '''
    fnames, scores = data
    batch = []
    stfts = []
    for aud in fnames:
        segments = segment(aud)[:-1]
        for seg in segments:
            clean = torch.tensor(10*np.log10(librosa.stft(seg, n_fft=512)))
            stfts.append(clean)
        batch.append(stfts)
        stfts = []
    return {"audio":torch.stack(batch), "score": torch.stack(torch.tensor(scores))}


class Data(data.Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.fnames = self.data['path']
        self.scores = self.data['MOS']

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.fnames[idx], self.scores[idx])
        return sample