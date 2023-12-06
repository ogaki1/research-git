
import numpy as np
from einops import rearrange
import torch
from torch import utils
import os
from define import *
import pandas as pd
from sklearn.model_selection import train_test_split

def one_hot_encode(labels, num_classes):
    one_hot = torch.zeros(labels.size(0), num_classes)
    for i in range(labels.size(0)):
        #print(labels[i])
        one_hot[i][labels[i]-1] = 1
    return one_hot

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, frame, label):
        self.video_acc_data = frame
        self.labels = label
    
    def __len__(self):
        return len(self.video_acc_data)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.video_acc_data[idx], dtype=torch.float32)
        Y = torch.tensor(self.labels[idx], dtype=torch.int8)
        #one_hot = torch.nn.functional.one_hot(Y,num_classes)
        #print(one_hot)
        #Y = label_to_one_hot(Y,num_classes)
        return X, Y

def sleep_data_to_numpy_array(dir_path):
    df = pd.DataFrame()
    path = os.listdir(dir_path)
    for i in path:
        df_tmp = pd.read_csv(os.path.join(dir_path,i))
        df = pd.concat([df,df_tmp])

    data_np = df.to_numpy()
    d = data_np[0:5].reshape([1,5,68])
    for i in range(1, int(len(df)/5)):
        d_tmp = data_np[5*i:5*(i+1)].reshape([1,5,68])
        d = np.append(d,d_tmp,axis=0)

    train, valid = train_test_split(d,shuffle=True, random_state=423504)

    #学習データ
    train_video_acc = train[0,:,0:67]
    train_labels = train[0,0,67]
    for i in range(1, len(train)):
        train_video_acc = np.append(train_video_acc,train[i,:,0:67],axis=0)
        train_labels = np.append(train_labels, train[i,0,67])
    #検証データ
    valid_video_acc = valid[0,:,0:67]
    valid_labels = valid[0,0,67]
    for i in range(1, len(valid)):
        valid_video_acc = np.append(valid_video_acc,valid[i,:,0:67],axis=0)
        valid_labels = np.append(valid_labels, valid[i,0,67])
    return train_video_acc.reshape([int(len(train_video_acc)/5),5,1,67]),train_labels,valid_video_acc.reshape([int(len(valid_video_acc)/5),5,1,67]),valid_labels
