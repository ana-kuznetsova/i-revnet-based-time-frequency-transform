import numpy as np
import os
import pandas as pd
import random
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


###### DATA PIPELINE ######

noisy_fnames = pd.read_csv(os.path.join(csv_dir, 'noisy-train.csv'))['path']
clean_fnames = pd.read_csv(os.path.join(csv_dir, 'clean-train.csv'))['path']
fnames = [(i, j) for i, j in zip(noisy_fnames, clean_fnames)]

shuffled_data = random.sample(fnames, k=len(fnames))[:int(len(fnames)*frac)]
noisy_fnames, clean_fnames = zip(*shuffled_data)

noisy_fnames = list(noisy_fnames)
clean_fnames = list(clean_fnames)


######## MODEL DEFINITION #################
estClean = modelDifinition.iRevNetMasking(layerNum, filt, initPad, maskEstimator).cuda(deviceNum)
optimizer = optim.Adam(estClean.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08)
lossFunc = eval('myLF.'+lossMode)


for param in estClean.parameters():
    nn.init.normal_(param, 0.0, 0.001)


######### K-FOLD CROSS VALIDATION #########
kf = KFold(n_splits=5, random_state=56, shuffle=True)
X_num = np.arange(len(noisy_fnames))

for train_index, test_index in kf.split(X_num):
    ### LOAD DATA ###
    #print("TRAIN:", train_index, "TEST:", test_index)
    print(type(train_index))
    train_noisy = noisy_fnames[train_index]
    train_clean = clean_fnames[train_index]

    print(train_noisy[0], train_clean[0])
