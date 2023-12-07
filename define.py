#data selection
img_d = True
acc_d = True
#parameter
total_frames = 5
frame_height = 8
frame_width = 8
frame_acc = 3
batch_size = 25
middle_layer = 512

model = 'LSTM'
#model = 'SimpleRNN'
#model = 'TimeSformer'

train = 'xstart'

rnn_input = 524 #524, 12, 512
hidden_size = 8
num_layers = 2
num_classes = 4
epoch_num = 50

