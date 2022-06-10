
import torch
import torch.nn as nn
import configuration as cfg
import numpy
import torch.nn.functional as F
class VRKINNN(nn.Module):
    def __init__(self,BATCH_SIZE ,num_of_filters1 ,num_of_filters2 ,karnel1 ,stride1 ,karnel2,stride2,drop_out1 ,drop_out2 ,output_fc1 ,output_fc2 ,hidden_size_lstm):
        super().__init__()

    # Conv2d(input_channels, output_channels, kernel_size)
        self.conv1 = nn.Conv2d(1, num_of_filters1, karnel1, stride=stride1)
        self.conv2 = nn.Conv2d(num_of_filters1, num_of_filters2, karnel2, stride=stride2)
        self.bn1 = nn.BatchNorm2d(num_of_filters1)
        self.bn2 = nn.BatchNorm2d(num_of_filters2)
        self.dropout1 = nn.Dropout(drop_out1)
        self.dropout2 = nn.Dropout(drop_out2)
        self.lstm = nn.LSTM(input_size=output_fc1, hidden_size=hidden_size_lstm)
        self.fc1 = nn.Linear(int(((((((450 - karnel1) / stride1) + 1) / 2)- karnel2 ) / stride2) + 1) * num_of_filters2, output_fc1)
        self.fc2 = nn.Linear(hidden_size_lstm, output_fc2)
        self.fcOut = nn.Linear(output_fc2, 1)
        self.sigmoid = nn.Sigmoid()

    def convs(self, x):
        # out_dim = in_dim - kernel_size + 1
        # 10,2, 6, 450
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        # 10,10, 2, 150
        x = F.max_pool2d(x, (1, 2))
        # 10,10,2,75
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        # 10,10,1,74
        return x

    def forward(self, x1, x2):

        x1P = x1[:, 0, None, :, :]
        x2P = x2[:, 0, None, :, :]
        x1P = self.convs(x1P)
        x1P = x1P.view(x1P.size(0), x1P.size(1) * 1 * x1P.size(3))
        x1P = self.fc1(x1P)
        x2P = self.convs(x2P)
        x2P = x2P.view(x2P.size(0), x2P.size(1) * 1 * x2P.size(3))
        x2P = self.fc1(x2P)
        x1 = x1P
        x1 = x1.view(-1, 1, x1.size(1))
        x2 = x2P
        x2 = x2.view(-1, 1, x2.size(1))
        x1 = self.lstm(x1)
        x2 = self.lstm(x2)
        x1 = x1[0][:,-1,:]
        x2 = x2[0][:, -1, :]
        x1 = self.fc2(x1)
        x2 = self.fc2(x2)
        x = torch.abs(x1 - x2)
        x = self.sigmoid(self.fcOut(x))


        return x.to(torch.float64)