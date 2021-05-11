import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import librosa
import random
import sys


from modules import takeModules as tm
from modules import myLossFunction as myLF
from iRevNet import modelDifinition


class Data(data.Dataset):
    def __init__(self, clean_paths, noisy_paths):
        self.clean_paths = clean_paths
        self.noisy_paths = noisy_paths
        
    def __len__(self):
        return len(self.clean_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.clean_paths[idx], self.noisy_paths[idx])

def collate_fn(data):
    clean_path = data[0]
    noisy_path = data[1]
    clean, _ = librosa.core.load(clean_path, sr=16000)
    noisy, _ = librosa.core.load(noisy_path, sr=16000)
    res = np.concatenate([clean, noisy])
    return torch.tensor(res)

def crossvalid(model=None,criterion=None,optimizer=None,dataset=None,k_fold=5):
    
    train_score = pd.Series()
    val_score = pd.Series()
    
    total_size = len(dataset)
    fraction = 1/k_fold
    seg = int(total_size * fraction)
    # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset 
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size

        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))
        
        train_set = torch.utils.data.dataset.Subset(dataset, train_indices)
        val_set = torch.utils.data.dataset.Subset(dataset, val_indices)
        

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=None, shuffle=True)

        val_loader = torch.utils.data.DataLoader(val_set, batch_size=None, shuffle=True)
        train_acc = train(res_model,criterion,optimizer,train_loader,epoch=1)
        train_score.at[i] = train_acc
        val_acc = valid(res_model,criterion,optimizer,val_loader)
        val_score.at[i] = val_acc
    
    return train_score,val_score

def init_model(layerNum=6, filt ='UNet5SpecNorm', 
              maskEstimator='binary', lossMode='SDR', 
              lr_init=0.0001, red=4, deviceNum=1):
    initPad=red-1
    estClean = modelDifinition.iRevNetMasking(layerNum, filt, initPad, maskEstimator).cuda(deviceNum)
    optimizer = optim.Adam(estClean.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08)
    lossFunc = eval('myLF.'+lossMode)
    
    for param in estClean.parameters():
        nn.init.normal_(param, 0.0, 0.001)

    return estClean, optimizer, lossFunc


def train(model, optimizer, 
         criterion, train_loader, 
         val_loader=None, maxEpoch=500, 
         batchSize=16, speechLen=2**15, deviceNum=1):
    for epoch in range(1, maxEpoch+1):
        sumLoss  = 0.0
        sumSDR = 0.0

        for setNum in range(len(train_loader)):
            sys.stdout.write('\repoch: '+str(epoch)+' TrnSet: '+str(setNum+1)+'/'+str(len(train_loader))) 
            sys.stdout.flush()
            
            batchNum = len(train_loader)//batchSize
            for utter in range(batchNum):
                print(utter)
                optimizer.zero_grad()
                s = torch.from_numpy(np.array([])).float().reshape((0, speechLen)).cuda(deviceNum)
                x = torch.from_numpy(np.array([])).float().reshape((0, speechLen)).cuda(deviceNum)

                for bs in range(batchSize):
                    stmp = trainMiniSet[perm2[bs+utter*batchSize]][0]
                    xtmp = trainMiniSet[perm2[bs+utter*batchSize]][1]
                    
                    if len(stmp)>speechLen:
                        st = np.random.randint(len(stmp)-speechLen)
                        end = st+speechLen
                        stmp = stmp[st:end]
                        xtmp = xtmp[st:end]
                    else:
                        zLen = speechLen - len(stmp)
                        tmpPad = torch.zeros((zLen)).cuda(deviceNum)
                        stmp = torch.cat((stmp, tmpPad), 0)
                        xtmp = torch.cat((xtmp, tmpPad), 0)
                    
                    stmp =  stmp.unsqueeze(0)
                    xtmp =  xtmp.unsqueeze(0)
                    s = torch.cat((s,stmp), 0)
                    x = torch.cat((x,xtmp), 0)
                y, _, _ = estClean(x)
                loss = lossFunc(s, x, y, smos, xmos)
                loss.backward()
                optimizer.step()
                
                sumLoss += loss.detach().cpu().numpy()
                    
        sys.stdout.write('\n')
        
    print("time/epoch(Train):"+str(time.time() - start))
    print("avg. loss:"+str(sumLoss/batchNum))
    trainLoss= np.append(trainLoss, sumLoss/batchNum)
    wandb.log({"loss": sumLoss/batchNum})
    
    if valRatio !=0:
        start = time.time()
        for param in estClean.parameters():
            param.requires_grad = False
        sumLoss_val  = 0.0
        sumSDR_val = 0.0
        perm1_val = np.random.permutation( len(validData) )
        for setNum in range( len(validData) ):
            sys.stdout.write('\repoch: '+str(epoch)+' ValSet: '+str(setNum+1)+'/'+str(len(validData))) 
            sys.stdout.flush()
            validMiniSet, valScoresMiniSet = tm.list_to_gpu( validData[ perm1_val[setNum] ], deviceNum )
            perm2_val = np.random.permutation( len(validMiniSet) )
            batchNum_val = len(validMiniSet)//batchSize
            for utter in range(batchNum_val):
                optimizer.zero_grad()
                s_val = torch.from_numpy(np.array([])).float().reshape((0, speechLen)).cuda(deviceNum)
                x_val = torch.from_numpy(np.array([])).float().reshape((0, speechLen)).cuda(deviceNum)
                for bs in range( batchSize ):
                    stmp = validMiniSet[perm2_val[bs+utter*batchSize]][0]
                    xtmp = validMiniSet[perm2_val[bs+utter*batchSize]][1]
                    smos = valScoresMiniSet[perm2_val[bs+utter*batchSize]][0]
                    xmos = valScoresMiniSet[perm2_val[bs+utter*batchSize]][1]
                    
                    if len(stmp)>speechLen:
                        st = np.random.randint(len(stmp)-speechLen)
                        end = st+speechLen
                        stmp = stmp[st:end]
                        xtmp = xtmp[st:end]
                    else:
                        zLen = speechLen - len(stmp)
                        tmpPad = torch.zeros((zLen)).cuda(deviceNum)
                        stmp = torch.cat( (stmp, tmpPad), 0 )
                        xtmp = torch.cat( (xtmp, tmpPad), 0 )
                    
                    stmp =  stmp.unsqueeze(0)
                    xtmp =  xtmp.unsqueeze(0)
                    s_val = torch.cat( (s_val,stmp), 0)
                    x_val = torch.cat( (x_val,xtmp), 0)
                    
                s_val.detach()
                x_val.detach()
                y_val, _, _ = estClean(x_val)
                d_val = x_val-y_val
                n_val = x_val-s_val
                loss = lossFunc(s_val, x_val, y_val, smos, xmos)
                sumLoss_val += loss.detach().cpu().numpy()
                    
        sys.stdout.write('\n')
        
        print("time/epoch(Valid):"+str(time.time() - start))
        print("avg. loss:"+str(sumLoss_val/batchNum_val))
        validLoss= np.append(validLoss, sumLoss_val/batchNum_val)
        wandb.log({"val_loss": sumLoss_val/batchNum_val})
        for param in estClean.parameters():
            param.requires_grad = True
        
    if testFlag == 0: 
        print('save DNN at epoch '+str(epoch))
        torch.save(estClean.state_dict(), fileName+'_ep'+str(epoch))
    else:
        print('test mode (do not save)') 

    train_score,val_score = crossvalid(res_model,criterion,optimizer,dataset=tiny_dataset)



#####main func######

clean_path = '/nobackup/anakuzne/data/COSINE-orig/clean-train'
noisy_path = '/nobackup/anakuzne/data/COSINE-orig/noisy-train'

fclean = os.listdir(clean_path)
cnames = [os.path.join(clean_path, i) for i in fclean]

fmix = os.listdir(noisy_path)
xnames = [os.path.join(noisy_path, i) for i in fmix]

dataset = Data(clean_paths=cnames,
               noisy_paths=xnames)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=True, collate_fn=collate_fn)
model, optimizer, criterion = init_model()

train(model, optimizer, criterion, train_loader)