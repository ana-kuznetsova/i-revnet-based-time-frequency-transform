# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:58:16 2020

@author: acoust
"""
import os, sys, glob
import numpy as np
from matplotlib import pylab as plt
import pandas as pd
from tqdm import tqdm
from pypesq import pesq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append('../')
from modules import takeModules as tm
from iRevNet import modelDifinition


def calc_pesq(csv, clean_path, enhanced_path):
    df = pd.read_csv(csv)
    names = df['path'].values
    scores = []
    for f in tqdm(names):
        clean_speech, fs = librosa.load(os.path.join(clean_path, f),sr=16000)
        enhanced_speech, fs = librosa.load(os.path.join(enhanced_path, f), sr=16000)
        enhanced_speech = 10*(enhanced_speech/np.linalg.norm(enhanced_speech))
        scores.append(pesq(clean_speech, enhanced_speech, 16000))
    return np.mean(np.array(scores))


##################################################################################
# flag
deviceNum  = 1

##################################################################################
# exp. param
layerNum = 6


filt = 'UNet5SpecNorm'
#filt = 'LinearNoBiasUNet5SpecNorm'

red = 4


maskEstimator = 'binary'
#maskEstimator = 'UNet5Sigmoid'

lossMode = 'SDR'




# training data directory
cleanDir  = '/nobackup/anakuzne/data/COSINE-orig/clean-train'
noisyDir  = '/nobackup/anakuzne/data/COSINE-orig/noisy-train'
csv_dir = '/nobackup/anakuzne/data/COSINE-orig/csv/'

# save dnn directory
dnn_dir  = '/nobackup/anakuzne/models/i-rev-net-mos/' 
if(os.path.isdir(dnn_dir)==False):
    os.mkdir(dnn_dir)
    
# train parameter
speechPerSet = 2048
batchSize = 16
Log_reg = 10**(-6)
valRatio = 0.1
speechLen = 2**15

maxEpoch = 500


initPad=red-1
##################################################################################
'''
saveName = \
'iRevNet_L'+str(layerNum)+\
'R'+str(initPad+1)+\
'_'+filt+\
'_'+maskEstimator+\
'_'+lossMode+\
'_bs'+str(batchSize)+\
'_bpl'+str(speechLen)+\
'_vr'+str(valRatio)\
+'_ep'+str(maxEpoch)
'''

fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

res_scores = []

for frac in fractions:
    print('Testing fraction ', frac)
    saveName = 'iRevNet_L6R4_UNet5SpecNorm_binarySDR_bs16_bpl32768_vr0.1COSINE_frac_'+str(frac)+'_ep500'
    fileName = os.path.join(dnn_dir, saveName)

    testDir = '/nobackup/anakuzne/models/i-rev-net-mos/test_out/'
    if(os.path.isdir(testDir)==False):
                os.mkdir(testDir)


    condDir = os.path.join(testDir, saveName)
    if(os.path.isdir(condDir)==False):
        os.mkdir(condDir)
            

##################################################################################


    estClean = modelDifinition.iRevNetMasking( layerNum, filt, initPad, maskEstimator).cuda(deviceNum)
    estClean.load_state_dict(torch.load(fileName))


    #sdataFns  = glob.glob(cleanDir + "/*.wav")
    #xdataFns  = glob.glob(noisyDir + "/*.wav")
    sdataFns = pd.read_csv(os.path.join(csv_dir, 'clean-test.csv'))['path'].values
    xdataFns = pd.read_csv(os.path.join(csv_dir, 'noisy-test.csv'))['path'].values

    sdataFns = [os.path.join(cleanDir, i) for i in sdataFns]
    xdataFns = [os.path.join(cleanDir, i) for i in sdataFns]
    testNum = len(sdataFns)

    for utter in range(testNum):
        sys.stdout.write('\rTestSet: '+str(utter+1)+'/'+str(testNum)) 
        sys.stdout.flush()
        s = torch.from_numpy(tm.wavread(sdataFns[utter])[0]).cuda(deviceNum)
        x = torch.from_numpy(tm.wavread(xdataFns[utter])[0]).cuda(deviceNum)
        sLen = len(s) 
        zp = speechLen - sLen%speechLen
        s = torch.cat( (s, torch.zeros(zp).cuda(deviceNum)), 0 ).unsqueeze(0)
        x = torch.cat( (x, torch.zeros(zp).cuda(deviceNum)), 0 ).unsqueeze(0)    
        y, phi, mask = estClean(x)
        y = y.detach()

        s = s[0][:sLen]
        x = x[0][:sLen]
        y = y[0][:sLen]
        
        saveFn = condDir+'/'+sdataFns[utter][len(cleanDir)+1:]
        tm.wavwrite(saveFn, y.cpu().numpy(), 16000)
    
    sys.stdout.write('\n')
    print('Finised inference...')
    print('Calculating PESQ...')

    frac_pesq = calc_pesq('/nobackup/anakuzne/data/COSINE-orig/csv/clean-test.csv',
                           '/nobackup/anakuzne/data/COSINE-orig/clean-train/',
                           '/nobackup/anakuzne/models/i-rev-net-mos/test_out/' )
    print("Fraction:", frac, "PESQ:", frac_pesq)
    res_scores.append(frac_pesq)
np.save('/nobackup/anakuzne/models/i-rev-net-mos/pesq_frac.npy', np.array(res_scores))
