import sys
import os
from model import Model

import numpy as np
import yaml

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse 

import collections
import os
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
from joblib import Parallel, delayed



###### For PartialSpoof DATSET  ########
# modified from https://github.com/eurecom-asp/rawnet2-antispoofing
ASVFile = collections.namedtuple('ASVFile',
    ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])


class Dataset(Dataset):
    def __init__(self, data_path=None, label_path=None, transform=None, is_train=True, is_eval=False, track='PARTIAL'):
        self.data_path_root = data_path
        self.label_path = label_path
        self.track = track
        self.is_eval = is_eval
        self.transform = transform
        
        if self.track =='FULL':
            if self.is_eval:
                self.sysid_dict = {
                '-': 0,  # bonafide speech
                'A07': 1,
                'A08': 2, 
                'A09': 3, 
                'A10': 4, 
                'A11': 5, 
                'A12': 6,
                'A13': 7, 
                'A14': 8, 
                'A15': 9, 
                'A16': 10, 
                'A17': 11, 
                'A18': 12,
                'A19': 13,    
            }
            else:
                self.sysid_dict = {
                '-': 0,  # bonafide speech         
                'A01': 1, 
                'A02': 2, 
                'A03': 3, 
                'A04': 4, 
                'A05': 5, 
                'A06': 6,        
            }
    
            self.sysid_dict_inv = {v:k for k,v in self.sysid_dict.items()}
            print('sysid_dict_inv',self.sysid_dict_inv)
    
            self.dset_name = 'eval' if is_eval else 'train' if is_train else 'dev'
            print('dset_name',self.dset_name)
    
            self.label_fname = 'eval.trl' if is_eval else 'train.trn' if is_train else 'dev.trl'
            print('label_fname',self.label_fname)
    
            self.label_dir = os.path.join(self.label_path)
            print('protocols_dir',self.label_dir)
    

            self.prefix = 'ASVspoof2019_LA'
            self.audio_files_dir = os.path.join(self.data_path_root, '{}_{}'.format(
                self.prefix, self.dset_name), 'flac')
            print('audio_files_dir',self.audio_files_dir)
    
            self.label_fname = os.path.join(self.label_dir,
                'ASVspoof2019.LA.cm.{}.txt'.format(self.label_fname))
            print('label_file',self.label_fname)

        else: #partial
            self.sysid_dict = {
                '-': 0,  # bonafide speech
                'CON': 1,}
            
            self.sysid_dict_inv = {v:k for k,v in self.sysid_dict.items()}
            print('sysid_dict_inv',self.sysid_dict_inv)
    
            self.dset_name = 'eval' if is_eval else 'train' if is_train else 'dev'
            print('dset_name',self.dset_name)
    
            self.label_fname = 'eval.trl' if is_eval else 'train.trl' if is_train else 'dev.trl'
            print('label_fname',self.label_fname)
    
            self.label_dir = os.path.join(self.label_path)
            print('protocols_dir',self.label_dir)
    

            self.audio_files_dir = os.path.join(self.data_path_root, 'database_{}'.format(self.dset_name),
                                                'database','{}'.format(self.dset_name), 'con_wav')
            print('audio_files_dir',self.audio_files_dir)
    
            self.label_fname = os.path.join(self.label_dir,
                'PartialSpoof.LA.cm.{}.txt'.format( self.label_fname))
            print('label_file',self.label_fname)
      
            

        cache_fname = 'cache_{}_{}.npy'.format(self.track, self.dset_name)
        self.cache_fname = os.path.join(data_path, cache_fname)
            
        if os.path.exists(self.cache_fname):
            self.data_x, self.data_y, self.data_sysid, self.files_meta = torch.load(self.cache_fname)
            print('Dataset loaded from cache', self.cache_fname)
        else: 
            self.files_meta = self.parse_protocols_file(self.label_fname)
            data = list(map(self.read_file, self.files_meta))
            self.data_x, self.data_y, self.data_sysid = map(list, zip(*data))
            if self.transform:
                self.data_x = Parallel(n_jobs=5, prefer='threads')(delayed(self.transform)(x) for x in self.data_x)                          
            torch.save((self.data_x, self.data_y, self.data_sysid, self.files_meta), self.cache_fname)
        
    def __len__(self):
        self.length = len(self.data_x)
        return self.length
   
    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, y, self.files_meta[idx]
            
    def read_file(self, meta):   
        data_x, sample_rate = sf.read(meta.path)
        data_y = meta.key
        return data_x, float(data_y) ,meta.sys_id   

    def parse_line(self, line):
        tokens = line.strip().split(' ')
        if self.track == 'PARTIAL':
            return ASVFile(speaker_id=tokens[0],
                file_name=tokens[1],
                path=os.path.join(self.audio_files_dir, tokens[1] + '.wav'),
                sys_id=self.sysid_dict[tokens[3]],
                key=int(tokens[4] == 'bonafide')) #bonafide:1 spoof:0
        
        else:
            if self.is_eval == False:  # multi-label for train + validation
                if tokens[4] == 'bonafide':
                    label = 1
                elif tokens[3] == 'A01':
                    label = 0
                elif tokens[3] == 'A02':
                    label = 0
                elif tokens[3] == 'A03':
                    label = 0
                elif tokens[3] == 'A04':
                    label = 0
                elif tokens[3] == 'A05':
                    label = 2
                elif tokens[3] == 'A06':
                    label = 2
                return ASVFile(speaker_id=tokens[0],
                    file_name=tokens[1],
                    path=os.path.join(self.audio_files_dir, tokens[1] + '.flac'),
                    sys_id=self.sysid_dict[tokens[3]],
                    key=label)  #  bonafide:1 TTS:0 VC:2
            
        return ASVFile(speaker_id=tokens[0],
            file_name=tokens[1],
            path=os.path.join(self.audio_files_dir, tokens[1] + '.flac'),
            sys_id=self.sysid_dict[tokens[3]],
            key=int(tokens[4] == 'bonafide')) # => #bonafide:1 spoof:0 for evaluation

    def parse_protocols_file(self, label_fname):
        lines = open(label_fname).readlines()
        files_meta = map(self.parse_line, lines)
        return list(files_meta)



def self_collate(batch):  # for batch-loading
    sigs, labels, lengths = zip(*[(a, b, a.size()) for (a,b,_) in sorted(batch, key=lambda x: x[0].size(0), reverse=True)])

    max_len = sigs[0].size()
    if max_len[0] <= 32100:
        max_len = torch.Size([32100])
        
    sigs = [torch.cat((s, torch.zeros(max_len[0] - s.size(0))), 0) if s.size(0) != max_len else s for s in sigs]

    sigs = torch.stack(sigs, 0)

    labels = Tensor(labels)

    return sigs, labels


def train_epoch(data_loader, model, lr,optim, device):
    train_loss = 0
    num_correct = 0.0
    num_total = 0.0

    model.train()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    CE = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in tqdm(train_loader):
       
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        loss = CE(batch_out, batch_y)
        
        _, pred = batch_out.max(dim=1)
        num_correct += (pred == batch_y).sum(dim=0).item()
        train_loss += (loss.item() * batch_size)

        optim.zero_grad()
        loss.backward()
        optim.step()
        
       
    train_loss /= num_total
    train_acc = (num_correct/num_total)*100
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return train_loss, train_acc



def validation(data_loader, model, device):
    valid_loss = 0
    num_correct = 0.0
    num_total = 0.0
    num_FRR = 0
    num_real_audio = 0
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    for batch_x, batch_y in tqdm(data_loader):
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        # _, _, batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        
        num_real_audio += (batch_y == 0).sum(dim=0).item()
        for i in range(len(batch_y)):
            if batch_y[i] == 0 and batch_pred[i] != batch_y[i]:
                num_FRR += 1
        
        
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        
        
        batch_loss = criterion(batch_out, batch_y)
        valid_loss += (batch_loss.item() * batch_size)
    valid_loss /= num_total
    valid_acc = 100 * (num_correct / num_total)
    valid_FRR = 100 * (num_FRR / num_real_audio)
    
    return valid_loss, valid_acc, valid_FRR





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_path', type=str, help='Change it to the directory of ASVSPOOF2019 database', required=True)
    parser.add_argument('--protocols_path', type=str, help='Change it to the directory of ASVSPOOF2019 (LA) protocols', required=True)
    args = parser.parse_args()
    

    np.random.seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Parameter
    config = yaml.safe_load(open('model_config.yaml'))
    num_epochs = config['epoch']
    lr = config['lr']
    weight_decay = config['wd']  
    batch_size = config['batch_size']


    # Model Initialization
    model = Model(device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,9], gamma=0.1)

    
    # To save model
    tag = '{}_{}'.format(config['batch_size'],lr)
    model_save_path = os.path.join('SAVED_MODELS', tag)
    if os.path.exists(model_save_path)==False:
        os.makedirs(model_save_path)
    writer = SummaryWriter('logs/{}'.format(tag))


    # Data loading
    transform = transforms.Compose([
        lambda x: Tensor(x)
    ])

    database_path = args.database_path
    label_path = args.protocols_path

    train_set = Dataset(data_path=database_path,label_path=label_path,is_train=True,transform=transform)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], collate_fn=self_collate, shuffle=True, drop_last=True)

    validate_set = Dataset(data_path = database_path,label_path = label_path,is_train=False, transform=transform)
    validate_loader = DataLoader(validate_set, batch_size=config['batch_size'], collate_fn=self_collate, shuffle=True)


    # Start training
    best_valid_acc = 90

    for epoch in range(num_epochs):
        running_loss, train_accuracy = train_epoch(train_loader,model, config['lr'], optimizer, device)
        scheduler.step()
        valid_loss, valid_acc, valid_FRR = validation(dev_loader, model, device)
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', valid_acc, epoch)
        writer.add_scalar('valid_loss', valid_loss, epoch)
        writer.add_scalar('valid_FRR', valid_FRR, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        print('\n{} - {:.4f} - {:.4f} - {:.4f} - {:.4f} - {:.4f}'.format(
	        epoch,running_loss, train_accuracy, valid_loss, valid_acc, valid_FRR))

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))

    writer.close()    