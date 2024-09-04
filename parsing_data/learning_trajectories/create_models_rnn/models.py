import torch
import torch
from torch import nn
from ncps.torch import CfC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class GRUModel1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nsteps=100, num_layers=1):
        super(GRUModel1, self).__init__()
        self.cell1 = nn.GRUCell(3, hidden_size)
        self.cell2 = nn.GRUCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.nsteps = nsteps

    def forward(self, z):
        nbatch = z.size(0)
        x = z[:,3:]
        h1 = torch.zeros(nbatch,self.hidden_size).to(device)
        h2 = torch.zeros(nbatch,self.hidden_size).to(device)
        seq = [z[:,3:]]
        for i in range(self.nsteps-1):
            h1 = self.cell1(x,h1)
            h2 = self.cell2(h1,h2)
            x_delta = torch.tanh(self.fc(h2))
            x = x+ 0.1*x_delta
            seq.append(x)

        out = torch.stack(seq,dim=0).permute(1,0,2)
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

