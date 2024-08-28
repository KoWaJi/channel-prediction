import torch.nn as nn
import torch.nn.functional as F
import torch

class CNNAugmentedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(CNNAugmentedLSTM, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.bn = nn.BatchNorm1d(num_features=8)
        self.fc = nn.Linear(hidden_size, 512)  # 512
        self.output = nn.Linear(512, 1)  # 512

    def forward(self, x):
        # 1D Conv expects: [batch, channels, seq_len]
        x = self.conv1d(x)
        #batch nomorlization
        x = self.bn(x)
        # LSTM expects: [batch, seq_len, features]
        x = x.permute(0, 2, 1)
        x, (ht, ct) = self.lstm(x)
        
        # Use the last hidden state
        x = ht[-1]
        
        # Pass through the fully connected layers
        x = F.leaky_relu(self.fc(x))
        x = self.output(x)
        return x


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size_lstm, hidden_size_fc, output_size, num_layers, dropout_rate):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size_lstm, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_size_lstm, output_size)

        self.fc = nn.Linear(hidden_size_lstm, hidden_size_fc)
        # 输出层，将512维度映射到output_size维度
        self.output = nn.Linear(hidden_size_fc, output_size)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.output(out)
        return out


class AT_CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(AT_CNNLSTM, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.bn = nn.BatchNorm1d(num_features=8)
        # 添加自注意力层，假设使用单个注意力头，注意力头的数量可以根据需要进行调整
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 512)
        self.output = nn.Linear(512, 1)

    def forward(self, x):
        # 1D Conv
        assert not torch.isinf(x).any(), "Input contains inf"
        x = self.conv1d(x)
        if torch.isnan(x).any():
            raise ValueError("NaN detected in conv1d output")
        # Batch normalization
        x = self.bn(x)
        # Permute for LSTM
        x = x.permute(0, 2, 1)
        x, (ht, ct) = self.lstm(x)
        
        # 引入自注意力机制，注意：输入和输出的维度需要匹配
        x, _ = self.self_attention(x, x, x)
        
        # 使用LSTM的最后一个隐藏状态
        x = ht[-1]
        
        # 通过全连接层
        x = F.leaky_relu(self.fc(x))
        x = self.output(x)
        return x
