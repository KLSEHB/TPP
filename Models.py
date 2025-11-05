import torch
import torch.nn as nn


# LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        #LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True )

        #16
        self.fcR1 = nn.Linear(hidden_size, 16)
        self.fcR2 = nn.Linear(16, output_size)

        # self.fcR3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):

        lstm_out, _ = self.lstm(x)  # 获取LSTM的输出

        out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        # out = torch.mean(lstm_out, dim=1)  # 对所有时间步的输出取平均

        out = self.fcR1(out)
        out = torch.relu(out)
        out = self.fcR2(out)

        # out = self.fcR3(out)

        return out

# MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # x: [batch_size, T, L] → flatten → [batch_size, T*L]
        x = x.view(x.size(0), -1)
        return self.model(x)

# GRU
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.gru  = nn.GRU(input_size, hidden_size, batch_first=True )
        self.fcR1 = nn.Linear(hidden_size, 16)
        self.fcR2 = nn.Linear(16, output_size)


    def forward(self, x):
        gru_out, _ = self.gru (x) 
        out = gru_out[:, -1, :] 
        out = self.fcR1(out)
        out = torch.relu(out) 
        out = self.fcR2(out)

        return out

# GRU
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fcR1 = nn.Linear(hidden_size, 16)
        self.fcR2 = nn.Linear(16, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # RNN 计算
        out = rnn_out[:, -1, :]  # 取最后时间步
        out = self.fcR1(out)
        out = torch.relu(out)  
        out = self.fcR2(out)

        return out
    
#Transformer
class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=2, num_layers=1):
        super(Transformer, self).__init__()
        
        self.embedding = nn.Linear(input_size, hidden_size)  
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.fcR1 = nn.Linear(hidden_size, 16)
        self.fcR2 = nn.Linear(16, output_size)

        self.fcR3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        x = self.embedding(x)  

        out = self.transformer_encoder(x)  
        out = out[:, -1, :] 

        out = torch.relu(self.fcR1(out))
        out = self.fcR2(out)

        return out

#LL    
class LinearBaseline(nn.Module):
    def __init__(self, input_size, seq_len, output_size):
        super(LinearBaseline, self).__init__()
        # 拉平成一个向量
        self.fc = nn.Linear(input_size * seq_len, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.reshape(x.shape[0], -1)   # (batch, seq_len * input_size)
        out = self.fc(x)
        return out



        

