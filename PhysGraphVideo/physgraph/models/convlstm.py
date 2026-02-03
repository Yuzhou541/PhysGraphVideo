import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size, padding=padding)
        self.hid_ch = hid_ch
    def forward(self, x, state):
        h, c = state
        if h is None:
            B,C,H,W = x.shape
            h = torch.zeros(B, self.hid_ch, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros_like(h)
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, (h, c)

class ConvLSTM(nn.Module):
    def __init__(self, in_ch, hid_ch, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([ConvLSTMCell(in_ch if i==0 else hid_ch, hid_ch) for i in range(num_layers)])
    def forward(self, xs):
        T,B,C,H,W = xs.shape
        states = [(None, None)] * len(self.layers)
        outs = []
        for t in range(T):
            x = xs[t]
            new_states = []
            for li, cell in enumerate(self.layers):
                h,c = states[li]
                h, st = cell(x, (h,c))
                new_states.append(st)
                x = h
            states = new_states
            outs.append(x)
        return torch.stack(outs, dim=0)
