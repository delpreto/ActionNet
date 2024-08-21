import torch
import torch
from torch import nn
from ncps.torch import CfC




class LSTMVanilla(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMVanilla, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, pos_encoding):
        x = x.unsqueeze(1).repeat(1, 99, 1)
        x = torch.cat([x, pos_encoding], dim=2)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)  # Take only the output of the last sequence step
        return out


class GRUVanilla(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUVanilla, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, pos_encoding):
        x = x.unsqueeze(1).repeat(1, 99, 1)
        x = torch.cat([x, pos_encoding], dim=2)
        out, _ = self.gru(x)
        out = self.fc(out)
        return out


class NCPVanilla(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NCPVanilla, self).__init__()
        self.ncp = CfC(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, pos_encoding):
        x = x.unsqueeze(1).repeat(1, 99, 1)
        x = torch.cat([x, pos_encoding], dim=2)
        out, _ = self.ncp(x)
        out = self.fc(out)
        return out

