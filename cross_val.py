import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from modules import takeModules as tm
from modules import myLossFunction as myLF
from iRevNet import modelDifinition
import wandb


#################################################################################
#wandb.init(project='i-rev-net-verify', entity='anakuzne')
##################################################################################
# flag
testFlag = 0 # 0:full train mode 1:test mode (few files) 
deviceNum  = 1

layerNum = 6
filt = 'UNet5SpecNorm'
red = 4

maskEstimator = 'binary'
lossMode = 'SDR'

# save dnn directory
dnn_dir  = '/nobackup/anakuzne/models/i-rev-net-mos/'
if(os.path.isdir(dnn_dir)==False):
    os.mkdir(dnn_dir)

# training data directory

cleanDir  = '/nobackup/anakuzne/data/COSINE-orig/clean-train'
noisyDir  = '/nobackup/anakuzne/data/COSINE-orig/noisy-train'
csv_dir = '/nobackup/anakuzne/data/COSINE-orig/csv/'

speechPerSet = 2048
batchSize = 16
Log_reg = 10**(-6)
valRatio = 0.1
speechLen = 2**15

frac = 0.1

maxEpoch = 500
lr_init  = 0.0001

config = wandb.config
config.learning_rate = lr_init
config.max_epoch = maxEpoch
config.speech_per_set = speechPerSet
config.batch_size = batchSize
config.loss_mode = lossMode
config.maskEstimator = maskEstimator
config.filter = filt
config.frac = frac
config.dataset = 'COSINE'
config.mode = 'cross-val'


initPad= red-1

saveName = \
'iRevNet_L'+str(layerNum)+\
'R'+str(initPad+1)+\
'_'+filt+\
'_'+maskEstimator+lossMode+\
'_bs'+str(batchSize)+\
'_bpl'+str(speechLen)+\
'_vr'+str(valRatio) + 'COSINE' + '_frac_' + str(frac) + 'cross'

fileName = os.path.join(dnn_dir, saveName)
print(fileName)


X_num = np.arange(7000)

kf = KFold(n_splits=5, random_state=56, shuffle=True)
for train_index, test_index in kf.split(X_num):
    print("TRAIN:", train_index, "TEST:", test_index)
