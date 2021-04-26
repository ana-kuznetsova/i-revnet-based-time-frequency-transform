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
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data


# Dataloader for Training/CV Data
class AudioDataset(data.Dataset):

    def __init__(self, mix_flist, clean_flist, batch_size, sample_rate=16000):
        """
        Inputs:
            mix_flist:   mix data file list
            clean_flist: clean data file list, must be same length as mix_flist
            batch_size:  mini-batch size
        """

        # generate minibach infomations
        minibatch = []
        start = 0
        while True:
            end = min(len(mix_flist), start + batch_size)

            tmp_batch = [[i,j] for i,j in zip(mix_flist[start:end],clean_flist[start:end])] # concatenate two lists
            minibatch.append(tmp_batch)

            if end == len(mix_flist):
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
    batch = batch[0]
    mix_batch, clean_batch, mixfilenames, cleanfilenames = load_mini_batch(batch)

    # to cuda device
    # get batch of lengths of input sequences
    ilens = np.array([len(mix) for mix in mix_batch])
    max_len = np.max(ilens)

    # perform padding and convert to tensor
    pad_value = 0
    # N x T
    mix_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mix_batch], pad_value, max_len)
    ilens = torch.from_numpy(ilens)
    # N x T
    clean_pad = pad_list([torch.from_numpy(c).float()
                            for c in clean_batch], pad_value, max_len)

    return mix_pad, clean_pad, ilens, mixfilenames, cleanfilenames

# Utility functions
# Loading for mini-batch
def load_mini_batch(batch):
    """
    Each info include wav path and wav duration.
    Outputs:
        mixtures: a list containing B items, each item is T np.ndarray
        sources: a list containing B items, each item is T x C np.ndarray
        T varies from item to item.
    """
    mix_batch, clean_batch  = [], []
    mixfilenames, cleanfilenames = [], []

    # for each utterance
    for i in range(len(batch)):
        
        cur_mix_file   = batch[i][0] # mix audio filename
        cur_clean_file = batch[i][1] # clean audio filename
        
        mixfilenames.append(cur_mix_file)
        cleanfilenames.append(cur_clean_file)
        """
        print('mix filename')
        print(cur_mix_file)

        print('clean filename')
        print(cur_clean_file)
        """
        
        cur_mix, _   = librosa.load(cur_mix_file, sr=16000)
        cur_clean, _ = librosa.load(cur_clean_file, sr=16000)
        
        mix_batch.append(cur_mix)
        clean_batch.append(cur_clean)

    return mix_batch, clean_batch, mixfilenames, cleanfilenames

# Padding for mini-batch
# pad to the max length in a mini-batch for every file
def pad_list(xs, pad_value, max_len):

    n_batch = len(xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


if __name__ == "__main__":
    import soundfile as sf

    # filelist for mix and clean audios
    mix_flist_name = '../data/testing_audio_mixfile.npy'
    clean_flist_name = '../data/testing_audio_cleanfile.npy'

    mix_flist = np.load(mix_flist_name)
    clean_flist = np.load(clean_flist_name)
    
    batch_size = 24 # temp tesing usage

    # sample dataloader for testing set
    dataset = AudioDataset(mix_flist, clean_flist, int(batch_size))
    data_loader = AudioDataLoader(dataset, batch_size=1,
                                  num_workers=10)

    # Sample test on data_loader
    for i, batch in enumerate(data_loader):
        if i%50 == 0:
            mix_batch, ref_batch, ilens, mixfilenames, cleanfilenames = batch
            print(i)
            print(mix_batch.size())
            print(ref_batch.size())
            #print(ilens)

            print(mixfilenames)
            print(cleanfilenames)
            
            """
            # test write audios
            sf.write('{}_mix.wav'.format(i), mix_batch[0,:ilens[0]], 16000)
            sf.write('{}_ref.wav'.format(i), ref_batch[0,:ilens[0]], 16000)

            sf.write('{}_mix.wav'.format(i+1), mix_batch[7,:ilens[7]], 16000)
            sf.write('{}_ref.wav'.format(i+1), ref_batch[7,:ilens[7]], 16000)
            """