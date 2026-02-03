import torch
import torch.nn as nn
from .convlstm import ConvLSTM

class FiLM(nn.Module):
    def __init__(self, chan, film_dim):
        super().__init__()
        self.fc = nn.Linear(film_dim, chan * 2)
        self.chan = chan
    def forward(self, x, cond):
        gb = self.fc(cond)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1 + gamma) + beta

class Encoder(nn.Module):
    def __init__(self, in_ch, hid_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, hid_ch, 3, padding=1),
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, hid_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hid_ch, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, out_ch, 3, padding=1),
        )
    def forward(self, x): return self.net(x)

class VideoGenerator(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, film_dim=128, num_layers=2):
        super().__init__()
        self.enc = Encoder(in_channels, hidden_dim)
        self.lstm = ConvLSTM(hidden_dim, hidden_dim, num_layers=num_layers)
        self.dec = Decoder(hidden_dim, in_channels)
        self.film = FiLM(hidden_dim, film_dim)
    def forward(self, seq, cond_vec):
        T = seq.size(0)
        encs = []
        for t in range(T):
            e = self.enc(seq[t])
            e = self.film(e, cond_vec)
            encs.append(e)
        encs = torch.stack(encs, dim=0)
        hs = self.lstm(encs)
        outs = [self.dec(hs[t]) for t in range(T)]
        return torch.stack(outs, dim=0)
