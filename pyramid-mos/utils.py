import pandas as pd
from tqdm import tqdm 
import numpy as np
import librosa


fnames = pd.read_csv('/nobackup/anakuzne/data/COSINE-orig/csv/all.csv')['path']

MAXLEN = 0

for f in tqdm(fnames):
    f, _ = librosa.core.load(f, sr=16000)
    f = librosa.stft(f, n_fft=512)
    MAXLEN = max(MAXLEN, f.shape[1])

print("MAXLEN:", MAXLEN)