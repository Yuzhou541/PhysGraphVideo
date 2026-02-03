import torch
try:
    import lpips
except Exception:
    lpips = None
class LPIPSLoss:
    def __call__(self, a, b):
        if lpips is None:
            return torch.tensor(0.0, device=a.device)
        T,B,C,H,W = a.shape
        net = lpips.LPIPS(net="alex")
        a = a.permute(1,0,2,3,4).reshape(B*T, C, H, W)
        b = b.permute(1,0,2,3,4).reshape(B*T, C, H, W)
        return net(a,b).mean()
