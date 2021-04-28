import pandas as pd
from tqdm import tqdm 
import numpy as np
import librosa


fnames = pd.read_csv('/nobackup/anakuzne/data/COSINE-orig/csv/all.csv').sample(frac=1, random_state=42)['path']
train = fnames[:int(len(fnames)*0.8)]
test = fnames[int(len(fnames)*0.8):]

stack = None
MAXLEN = 0

for f in tqdm(train):
    f, _ = librosa.core.load(f, sr=16000)
    f = np.nan_to_num(np.abs(10*np.log10(librosa.stft(f, n_fft=512))))
    MAXLEN = max(MAXLEN, f.shape[1])
    print(MAXLEN)

print("MAXLEN:", MAXLEN)
'''
    if stack is None:
        stack = f 
    else:
        stack = np.concatenate((stack, f), axis=1)


print("Mean Train:", stack.mean(), "Std Train:", stack.std())

stack = None

for f in tqdm(test):
    f, _ = librosa.core.load(f, sr=16000)
    f = 10*np.log10(librosa.stft(f, n_fft=512))
    #MAXLEN = max(MAXLEN, f.shape[1])

    if stack is None:
        stack = f 
    else:
        stack = np.concatenate((stack, f), axis=1)

#print("MAXLEN:", MAXLEN)
print("Mean Test:", stack.mean(), "Std Test:", stack.std())
'''
