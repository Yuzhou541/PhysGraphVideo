import torch.nn as nn
class TextEncoder(nn.Module):
    def __init__(self, vocab_size=4096, film_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 128)
        self.mlp = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, film_dim))
    def forward(self, token_ids):
        x = self.emb(token_ids).mean(dim=1)
        return self.mlp(x)
