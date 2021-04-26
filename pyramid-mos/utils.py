import pandas as pd
from tqdm import tqdm 
import numpy as np
import librosa


fnames = pd.read_csv('/nobackup/anakuzne/data/COSINE-orig/csv/all.csv')['path']

all_ = None

for i, f in tqdm(enumerate(fnames)):
    f, _ = librosa.core.load(f, 16000)
    f = 10*np.log10(librosa.stft(f, n_fft=512))
    if i == 0:
        all_ = f
    else:
        all_ = np.concatenate((all_,f), axis=1)

print("mean:", all_.mean(), "std:", all_.std())