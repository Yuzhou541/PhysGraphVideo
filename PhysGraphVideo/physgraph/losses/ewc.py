import torch
class EWC:
    def __init__(self, model, lambda_=40.0):
        self.lambda_ = lambda_
        self.params = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p) for n,p in model.named_parameters() if p.requires_grad}
    def accumulate_fisher(self, model, loss):
        loss.backward(retain_graph=True)
        with torch.no_grad():
            for n,p in model.named_parameters():
                if p.grad is None or n not in self.fisher: continue
                self.fisher[n] += p.grad.detach()**2
    def normalize(self, steps=1):
        for n in self.fisher: self.fisher[n] /= max(1, steps)
    def penalty(self, model):
        pen = 0.0
        for n,p in model.named_parameters():
            if n not in self.params: continue
            pen += (self.fisher[n] * (p - self.params[n])**2).sum()
        return self.lambda_ * pen
