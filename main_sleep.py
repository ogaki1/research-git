from data import *
from modelLSTM import *
from timesformer_pytorch import TimeSformer
from torchinfo import summary
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.model_selection import train_test_split
from torcheval.metrics.functional import multiclass_confusion_matrix

data_folder = './dataset/ML_data'
f = open('./experiment.txt', 'a') #出力ファイル
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#f.write('1回目-------------------------------\n')

train_frames_array, train_labels , valid_frames_array, valid_labels = sleep_data_to_numpy_array(data_folder)
#print(f'train_frames_shape:{train_labels.shape}')
#print(f'test_frames_shape:{valid_frames_array.shape}')

TrainDataset = MyDataset(train_frames_array,train_labels)
TestDataset = MyDataset(valid_frames_array,valid_labels)

train_Loader = torch.utils.data.DataLoader(dataset=TrainDataset,batch_size=batch_size)
test_Loader = torch.utils.data.DataLoader(dataset=TestDataset,batch_size=batch_size)
print(f'train_Loader:{train_Loader}')
print(f'test_Loader:{test_Loader}')

'''
for i, data in enumerate(test_Loader):
    inputs, labels = data[0].to(device), data[1] # data は [inputs, labels] のリスト
    print(inputs.shape)
'''

assert model in ['LSTM', 'SimpleRNN', 'TimeSformer'], 'The model is incorrectly defined.'

#モデル定義
if model == 'LSTM':
    cnn_1 = CNN(image_d=img_d, accelerate_d=acc_d).to(device)
    lstm = LSTM(input_size=rnn_input, hidden_size=hidden_size, num_layers=num_layers).to(device)
    model_LSTM = ConvLSTM(cnn_1,lstm,num_classes,hidden_size).to(device)
    optimizer = optim.Adam(model_LSTM.parameters(), lr=0.001)
    summary(model_LSTM, input_size=(batch_size,total_frames,1,frame_width*frame_height+frame_acc),col_names=["output_size", "num_params"])
    #para = summary(cnn_1, input_size=(batch_size,total_frames,1,frame_width*frame_height+frame_acc),col_names=["output_size"])


elif model == 'SimpleRNN':
    cnn_2 = CNN(image_d=img_d, accelerate_d=acc_d).to(device)
    simple_rnn = SimpleRNN(input_size=rnn_input, hidden_size=hidden_size, num_layers=num_layers).to(device)
    model_SimpleRNN = ConvSimpleRNN(cnn_2,simple_rnn,num_classes,hidden_size).to(device)
    optimizer = optim.Adam(model_SimpleRNN.parameters(), lr=0.001)
    summary(model_SimpleRNN, input_size=(batch_size,total_frames,1,frame_width,frame_height),col_names=["output_size", "num_params"])


elif model == 'TimeSformer':
    model_timesformer = TimeSformer(
        dim = 32,
        image_size=frame_height,
        patch_size=4,
        num_frames=total_frames,
        num_classes=num_classes,
        depth = 2,
        heads = total_frames,
        dim_head= 16,
        attn_dropout = 0.2,
        ff_dropout = 0.2
    ).to(device)
    #optimizer = optim.Adam(model_timesformer.parameters())
    optimizer = optim.Adamax(model_timesformer.parameters(),lr=0.001)
    summary(model_timesformer, input_size=(batch_size,total_frames,1,frame_width,frame_height),col_names=["output_size", "num_params"])

#損失関数を定義
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = nn.CrossEntropyLoss()

#print(model_LSTM.state_dict())
assert train in ['start'], 'モデルのみを表示'

# 畳み込み演算の実行
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

'''
for param in model_LSTM.parameters():
  print(param)
  param = nn.init.normal_(param)
'''

min_loss = 999999999
print("training start")
for epoch in range(epoch_num):
    train_loss = 0.0
    train_acc = 0
    val_loss = 0.0
    val_acc = 0
    train_batches = 0
    val_batches = 0
    train_confusion_matrix = torch.zeros(5,5).to(device)
    val_confusion_matrix = torch.zeros(5,5).to(device)

    # 訓練モード
    if model == 'LSTM':
        model_LSTM.train()   
    elif model == 'SimpleRNN':
        model_SimpleRNN.train()
    elif model == 'TimeSformer':
        model_timesformer.train()

    for i, data in enumerate(train_Loader):   # バッチ毎に読み込む
        inputs, labels = data[0].to(device), data[1].to(device) # data は [inputs, labels] のリスト
        one_hot_label = one_hot_encode(labels,num_classes).to(device)
        #print(one_hot_label)

        # 勾配のリセット
        optimizer.zero_grad()
        #print(inputs[:,:,:,:].shape)
        # 順方向計算
        if model == 'LSTM': 
            outputs = model_LSTM(inputs[:,0:total_frames,:,:]) 
        elif model == 'SimpleRNN':
            outputs = model_SimpleRNN(inputs)
        elif model == 'TimeSformer':
            outputs = model_timesformer(inputs)
        #print(outputs.max(1)[1]+1)
        #print((outputs.max(1)[1]+1)== labels.to(device))
        loss = criterion(outputs, one_hot_label) # 損失の計算
        loss.backward()                     # 逆方向計算(勾配計算)
        optimizer.step()                    # パラメータの更新
        
 
        # 履歴の累積
        train_loss += loss.item()
        train_batches += 1
        train_acc += (outputs.max(1)[1]+1 == labels).sum().item()
        #print(train_confusion_matrix)
        train_confusion_matrix += multiclass_confusion_matrix(outputs.max(1)[1]+1,labels,5)

    #検証モード
    if model == 'LSTM':
        model_LSTM.eval()   
    elif model == 'SimpleRNN':
        model_SimpleRNN.eval()
    elif model == 'TimeSformer':
        model_timesformer.eval() 

    with torch.no_grad():
        for i, data in enumerate(test_Loader):
            inputs, labels = data[0].to(device), data[1].to(device) # data は [inputs, labels] のリスト
            one_hot_label = one_hot_encode(labels,num_classes).to(device)

            # 順方向計算
            if model == 'LSTM': 
                outputs = model_LSTM(inputs) 
            elif model == 'SimpleRNN':
                outputs = model_SimpleRNN(inputs)
            elif model == 'TimeSformer':
                outputs = model_timesformer(inputs)
            
            loss = criterion(outputs, one_hot_label) # 損失の計算
            # 履歴の累積
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1]+1 == labels).sum().item()    
            #print(outputs.max(1)[1]+1)          
            val_confusion_matrix += multiclass_confusion_matrix(outputs.max(1)[1]+1,labels,5)
    # 履歴の出力

    print('epoch %d train_loss: %.5f train_acc %.5f test_loss: %.5f test_acc: %.5f' %
           (epoch + 1,  train_loss/len(train_Loader.dataset), train_acc/len(train_Loader.dataset), val_loss/len(test_Loader.dataset), val_acc/len(test_Loader.dataset)),file=f)

    #print(len(train_Loader.dataset))
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc/len(train_Loader.dataset))
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc/len(test_Loader.dataset))
    #scheduler.step()

print(train_confusion_matrix)
print(val_confusion_matrix)
'''
f.write('train_loss_list='+str(train_loss_list)+'\n')
f.write('train_acc_list='+str(train_acc_list)+'\n')
f.write('val_loss_list='+str(val_loss_list)+'\n')
f.write('val_acc_list='+str(val_acc_list)+'\n')
f.write('train_confusion_matrix\n'+str(train_confusion_matrix[1:,1:])+'\n')
f.write('val_confusion_matrix\n'+str(val_confusion_matrix[1:,1:])+'\n')

#テキストファイルを閉じる
f.close()

ticks = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
# 学習曲線 (精度)
plt.figure(figsize=(8,6))
plt.plot(val_acc_list,label='val', lw=3, c='b',linestyle='-')
plt.plot(train_acc_list,label='train', lw=3, c='r',linestyle='--')
plt.title('学習曲線 (精度)')
plt.xticks(size=14)
plt.yticks(ticks=ticks,size=14)
plt.grid(lw=2)
plt.legend(fontsize=14)
plt.savefig("acc_accuracy.png")

plt.figure(figsize=(8,6))
plt.plot(val_loss_list,label='val', lw=3, c='b',linestyle='-')
plt.plot(train_loss_list,label='train', lw=3, c='r',linestyle='--')
plt.title('学習曲線 (loss)')
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.legend(fontsize=14)
plt.savefig("acc_loss.png")
'''















