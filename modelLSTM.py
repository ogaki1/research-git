import torch
import torch.nn as nn
from torchinfo import summary
import torch.optim as optim
from define import *

torch.manual_seed(423504)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #imageデータ
        self.batch_norm0 = nn.BatchNorm2d(1, momentum=0.8)

        self.batch_norm1 = nn.BatchNorm2d(32,momentum=0.8)
        self.batch_norm2 = nn.BatchNorm2d(16,momentum=0.8)
        self.batch_norm3 = nn.BatchNorm2d(32,momentum=0.8)
        self.batch_norm4 = nn.BatchNorm2d(32,momentum=0.8)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding='same')

        self.max_pool1 = nn.MaxPool2d(2,stride=2)
        self.max_pool2 = nn.MaxPool2d(2,stride=2)
        self.max_pool3 = nn.MaxPool2d(2,stride=2)
        self.max_pool4 = nn.MaxPool2d(2,stride=2)

        self.dropout1_2 = nn.Dropout2d(0.2)
        self.dropout3_4 = nn.Dropout2d(0.3)
    
        #image 初期値
        nn.init.normal_(self.conv1.weight)
        nn.init.normal_(self.conv1.bias)

        #accelerationデータ
        self.batch_norm0_acc = nn.BatchNorm1d(1, momentum=0.8)

        self.batch_norm1_acc = nn.BatchNorm1d(4, momentum=0.8)

        self.conv1_acc = nn.Conv1d(1,4,kernel_size=2,padding='same')

        self.max_pool1_acc = nn.MaxPool1d(2)
        self.dropout1_2_acc = nn.Dropout1d(0.2)

        #acc 初期化
        nn.init.normal_(self.conv1_acc.weight)
        nn.init.normal_(self.conv1_acc.bias)

        #共通
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
    def forward(self, x):
        #print(x.shape)
        # 入力テンソルの形状を (バッチサイズ * 時間ステップ数, チャンネル数, 画像高さ, 画像幅) に変形
        batch_size = x.shape[0]
        time_steps = x.shape[1]
        x_img = x[:,:,:,0:64].view(-1,1,8,8)
        #x_img = torch.reshape(x_img,(x_img.size(0)*x_img.size(1),1,8,8))
        x_acc = x[:,:,:,64:67].view(-1,1,frame_acc)
        #print(x_img.shape,x_acc.shape)
        #imageデータ

        x_img = self.batch_norm0(x_img)
        #print(x.shape)
        x_img = self.relu(self.conv1(x_img))
        x_img = self.batch_norm1(x_img)
        x_img = self.max_pool1(x_img)
        x_img = self.dropout1_2(x_img)
        '''
        #print(x.shape)
        x_img = self.relu(self.conv2(x_img))
        x_img = self.batch_norm2(x_img)
        x_img = self.max_pool2(x_img)
        x_img = self.dropout1_2(x_img)
        #print(x.shape)

        x_img = self.leaky_relu(self.conv3(x_img))
        x_img = self.batch_norm3(x_img)
        x_img = self.max_pool3(x_img)
        x_img = self.dropout3_4(x_img)
        #print(x.shape)

        x_img = self.leaky_relu(self.conv4(x_img))
        x_img = self.batch_norm4(x_img)
        x_img = self.max_pool4(x_img)
        x_img = self.dropout3_4(x_img)
        '''
        x_img = self.flatten(x_img)

        #print(x_img)
        #accelerationデータ
        x_acc = self.batch_norm0_acc(x_acc)
        x_acc = self.relu(self.conv1_acc(x_acc))
        x_acc = self.batch_norm1_acc(x_acc)
        #x_acc = self.max_pool1_acc(x_acc)
        #x_acc = self.dropout1_2_acc(x_acc)
        x_acc = self.flatten(x_acc)

        x = torch.cat([x_img,x_acc],dim=1)
        
        #x = x_img
        #x = x_acc
        #print(x.shape)
        #print(x.shape)
        # 出力テンソルの形状を元の形に戻す
        x = x.view(batch_size, time_steps, x.shape[-1])
        #print(x.shape)
        return x

# LSTMモデルの定義
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        #self.fc = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        #print(x.size(0))
        out, _ = self.lstm(x, (h0, c0))
        #out = self.fc(out[:, -1, :])
        out = out[:, -1, :]
        return out

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        #self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, h0)
        #out = self.fc(out[:, -1, :])
        out = out[:, -1, :]
        return out

class ConvLSTM(nn.Module):
    def __init__(self,cnn,lstm,num_classes,hidden_size):
        super(ConvLSTM, self).__init__()
        self.cnn = cnn
        self.lstm = lstm
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, middle_layer)
        self.fc2 = nn.Linear(middle_layer,self.num_classes)
        self.layernorm1 = nn.LayerNorm(middle_layer)
        self.layernorm2 = nn.LayerNorm(self.num_classes)

        nn.init.normal_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias)

    def forward(self,x):
        x = self.cnn(x)
        x = self.lstm(x)
        x = self.fc1(x)
        x = self.layernorm1(x)
        x = self.fc2(x)
        x = self.layernorm2(x)
   
        return x
    
class ConvSimpleRNN(nn.Module):
    def __init__(self,cnn,rnn,num_classes,hidden_size):
        super(ConvSimpleRNN, self).__init__()
        self.cnn = cnn
        self.rnn = rnn
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, middle_layer)
        self.fc2 = nn.Linear(middle_layer,self.num_classes)
        self.layernorm1 = nn.LayerNorm(middle_layer)

    def forward(self,x):
        x = self.cnn(x)
        x = self.rnn(x)
        x = self.fc1(x)
        x = self.layernorm1(x)
        x = self.fc2(x)

        return x

