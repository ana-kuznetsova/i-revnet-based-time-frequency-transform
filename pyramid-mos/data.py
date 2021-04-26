 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Last modified on 03/23/2021
@author: Zhuohuang Zhang @ Ph.D. Student at Indiana University Bloomington

"""

import os 
import json
import math
import numpy as np
import librosa
import pandas as pd

import torch
import torch.utils.data as data


# Dataloader for Training/CV Data
class AudioDataset(data.Dataset):

    def __init__(self, csv_path, batch_size, sample_rate=16000):
        """
        Inputs:
            mix_flist:   mix data file list
            clean_flist: clean data file list, must be same length as mix_flist
            batch_size:  mini-batch size
        """

        # generate minibach infomations
        df = pd.read_csv(csv_path)
        flist = df['path']
        mos_scores  = df['MOS']
        minibatch = []
        start = 0
        while True:
            end = min(len(flist), start + batch_size)

            tmp_batch = [[i,j] for i,j in zip(flist[start:end],mos_scores[start:end])] # concatenate two lists
            minibatch.append(tmp_batch)

            if end == len(flist):
                break
            start = end
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)

# Dataloader will get batch information from Dataset object (i.e., batch of filelist)
class AudioDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


def _collate_fn(batch):
    """
    Inputs:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Outputs:
        mix_pad: B x T, torch.Tensor
        refs_pad: B x C x T, torch.Tensor
    """
    # batch should be located in list
    assert len(batch) == 1

    # w/o zero padding
    #batch_scores = batch[1]

    batch = batch[0]

    aud_batch, aud_fnames_batch, mos_scores = load_mini_batch(batch)

    # to cuda device
    # get batch of lengths of input sequences
    ilens = np.array([len(aud) for aud in aud_batch])
    max_len = np.max(ilens)

    # perform padding and convert to tensor
    pad_value = 0
    # N x T
    print(aud_batch)
    #aud_pad = pad_list([aud for aud in aud_batch], pad_value, max_len)
    #ilens = torch.from_numpy(ilens)

    return aud_pad, ilens, aud_fnames_batch, mos_scores

# Utility functions
# Loading for mini-batch
def load_mini_batch(batch):
    aud_batch = []
    aud_filenames = []
    mos_scores = []

    # for each utterance
    for i in range(len(batch)):
        
        cur_file   = batch[i][0] # mix audio filename
        
        aud_filenames.append(cur_file)
        mos_scores.append(batch[i][1])

        cur_aud, _   = librosa.load(cur_file, sr=16000)
        cur_aud = librosa.stft(cur_aud, n_fft=512)
        print(cur_aud.shape)
        
        aud_batch.append(cur_aud)

    return aud_batch, aud_filenames, mos_scores

# Padding for mini-batch
# pad to the max length in a mini-batch for every file
def pad_list(xs, pad_value, max_len):
    pass


if __name__ == "__main__":
    import soundfile as sf

    batch_size = 24 # temp tesing usage

    # sample dataloader for testing set
    dataset = AudioDataset('/nobackup/anakuzne/data/COSINE-orig/csv/all.csv', int(batch_size))
    data_loader = AudioDataLoader(dataset, batch_size=1,
                                  num_workers=10)
    
    
    # Sample test on data_loader
    for i, batch in enumerate(data_loader):
        if i%50 == 0:
            aud_pad, ilens, aud_fnames_batch, batch_scores = batch
            
            