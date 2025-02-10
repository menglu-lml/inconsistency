import collections
import librosa, librosa.display
import soundfile as sf

import torch
from torch import Tensor
from torchvision import transforms
from torchaudio import transforms as audioTran
from torch.utils.data import Subset
from torch import nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Model

import os
import math
import yaml
import copy



class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        self.device = device
        pre_trained_model_id = 'facebook/wav2vec2-xls-r-300m'
        self.processing = Wav2Vec2FeatureExtractor.from_pretrained(pre_trained_model_id)
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(pre_trained_model_id)
        self.out_dim = 1024
    
        return

    def extract_feat(self, input_data): 
        emb = self.processing(input_data,sampling_rate=16000,padding=True,return_tensors="pt").input_values[0].to(self.device)
        embb = self.wav2vec2_model(emb).last_hidden_state # [batch, 201, 1024] 
        del emb
        torch.cuda.empty_cache()
        return embb


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResNet(nn.Module):
    def __init__(self,f_dim):
        super(ResNet, self).__init__()
        
        self.in_planes = f_dim

        self.conv1 = nn.Conv1d(self.in_planes , self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_planes)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(self.in_planes, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm1d(self.in_planes)
        self.bn3 = nn.BatchNorm1d(self.in_planes)
        self.act2 = nn.ReLU()
        self.se = SELayer(self.in_planes)


    def forward(self, x):
        shortcut = x
        
        x = self.act1(self.bn1(x))
        x = self.conv1(x) 
       
        x = self.act2(self.bn2(x))
        x = self.conv2(x) 
        

        x = self.se(x)
        x += shortcut

        return x



class Model(nn.Module):
    def __init__(self,device, class_num = 2):
        super().__init__()
        self.device = device
      ####
        # create network wav2vec 2.0
        ####
        self.class_num = class_num
        self.ssl_model = SSLModel(self.device)

        
        self.emb_mid_len = 256
        self.emb_dim = 64
        self.channel_num = 32
        self.divide = 8

        
        self.fc1 = nn.Linear(self.ssl_model.out_dim, self.emb_mid_len)
        self.bn1 = nn.BatchNorm1d(self.emb_mid_len)
        self.relu1 = nn.LeakyReLU()
        
        
        self.fc2 = nn.Linear(self.emb_mid_len, self.emb_dim)
        
        
        self.ada_pool = nn.AdaptiveAvgPool2d((200, self.ssl_model.out_dim))
        self.resnet_2 = ResNet(self.emb_dim)
        

        self.conv1 = nn.Conv2d(1, 1, (1,3), padding=(0,1))
        self.conv2 = nn.Conv2d(1, self.channel_num, (3,1), padding=(1,0))

        self.downsample = nn.Conv2d(self.channel_num,self.channel_num // self.divide, 1)
        self.relu3 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(self.channel_num// self.divide, self.channel_num// self.divide, (5,1), padding=(1,0))
        self.upsample = nn.Conv2d(self.channel_num// self.divide,self.channel_num, 1)
        
        
        self.sigmoid = nn.Sigmoid()
        self.conv4 = nn.Conv2d(self.channel_num,1, 1)
        self.dense1 = nn.Linear(self.emb_dim, 1)
        self.dense2 = nn.Linear(200, 2)


    
    def forward(self, x):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        emb = self.ssl_model.extract_feat(x.squeeze(-1))  #feature embeddings  (bs,x,1024)
        x = self.ada_pool(emb)  #feature embeddings  (bs,200,1024)

        x = self.fc1(x)#(bs,200,256)
        
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn1(x)
        x = x.view(org_size)  #--
        x = self.relu1(x)

        
        x = self.fc2(x)
        x = x.transpose(1,2)
        x = self.resnet_2(x) 
        x = x.transpose(1,2)  



        batch_size = x.shape[0]
        seq_len = x.shape[1]
        embb = x.shape[2]  
        
        x_ = x.view(batch_size,1, seq_len,embb)

        x_tplus = self.conv1(x_)

        D = x_tplus[:,:,1:,:]  - x_[:,:,:-1,:]   
        D = F.pad(D, (0,0,0,1), "constant", 0)  
        D = D.view(batch_size,1, seq_len,embb)

        
        
        path_1 = self.conv2(D) 
        
        path_2 = self.downsample(path_1)
        path_2 = self.relu3(path_2)
        path_2 = self.conv3(path_2) 
        path_2 = self.upsample(path_2)
        
        
        D = self.conv4(path_1 + path_2).squeeze() 
        D = self.sigmoid(D)

      
        x = torch.mul(D, x)   #f meature embeddings  (bs,200,128)

        x = self.dense1(x)
        x = self.dense2(x.squeeze()) 
            

        return x
