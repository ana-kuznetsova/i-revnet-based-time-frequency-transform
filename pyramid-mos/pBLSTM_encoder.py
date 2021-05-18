#!/usr/bin/env python
# coding: utf-8

import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import nn
import torch


class pBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=1, bidirectional=True)
    def forward(self, x):
        return self.blstm(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, value_size=128,key_size=128):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            bidirectional=True)
        self.pBLSTM1 = pBLSTM(2*hidden_dim, hidden_dim)
        self.pBLSTM2 = pBLSTM(2*hidden_dim, hidden_dim)
        self.pBLSTM3 = pBLSTM(2*hidden_dim, hidden_dim)
        self.key_network = nn.Linear(hidden_dim*2, value_size)
        self.value_network = nn.Linear(hidden_dim*2, key_size)
  
    def forward(self, x, lens):
        x = torch.transpose(1, -1)
        print("X", x.shape)
        rnn_inp = pack_padded_sequence(x, lengths=lens, enforce_sorted=True, batch_first=True)
        print("rnn inp:", rnn_inp.shape)

        outputs, _ = self.lstm(rnn_inp)
        linear_input, _= pad_packed_sequence(outputs)
        print("IN shape:", linear_input.shape)

        for i in range(3):
            if linear_input.shape[0]%2!=0:
                linear_input = linear_input[:-1,:,:]
            outputs = torch.transpose(linear_input, 0, 1)
            outputs = outputs.contiguous().view(outputs.shape[0], outputs.shape[1]//2, 2, outputs.shape[2])
            outputs = torch.mean(outputs, 2)
            outputs = torch.transpose(outputs,0,1)
            lens=lens//2
            rnn_inp = pack_padded_sequence(outputs, lengths=lens, enforce_sorted=True, batch_first=True)
            if i==0:
                outputs, _ = self.pBLSTM1(rnn_inp)
            elif i==1:
                outputs, _ = self.pBLSTM2(rnn_inp)
            else:
                outputs, _ = self.pBLSTM3(rnn_inp)
            linear_input, _ = pad_packed_sequence(outputs)
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)

        return keys, value, lens

#######################################
from data import Data, collate_custom
import torch.utils.data as data

csv_path = '/nobackup/anakuzne/data/COSINE-orig/csv/all.csv'
dataset = Data(csv_path, mode='train')
loader = data.DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=collate_custom)

input_dim = 601
hidden_dim = 128
device = 'cuda:1'

encoder = Encoder(input_dim, hidden_dim)
encoder.to(device)


for batch in loader:
    audio = batch['aud'].to(device)
    lens = batch['lens']
    mos = batch['score'].to(device)
    K, V, _ = encoder(audio, lens)
    print(K.shape)
    