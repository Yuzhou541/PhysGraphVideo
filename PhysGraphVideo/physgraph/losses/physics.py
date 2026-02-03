import torch
def second_order_accel_loss(seq):
    if seq.size(0) < 3: return torch.tensor(0.0, device=seq.device)
    a = seq[2:] - 2*seq[1:-1] + seq[:-2]
    return (a**2).mean()
def divergence_like_smoothness(seq):
    dx = seq[..., 1:, :, :] - seq[..., :-1, :, :]
    dy = seq[..., :, 1:, :] - seq[..., :, :-1, :]
    dt = seq[1:] - seq[:-1]
    return (dx.abs().mean() + dy.abs().mean() + dt.abs().mean())/3.0
def boundary_penalty(seq):
    T,B,C,H,W = seq.shape
    mask = torch.zeros(1,1,H,W, device=seq.device)
    mask[:,:,0,:]=1; mask[:,:,-1,:]=1; mask[:,:,:,0]=1; mask[:,:,:,-1]=1
    return (seq * mask).abs().mean()
def physics_loss(seq, w):
    loss = 0.0
    if w.get("accel",0)>0: loss += w["accel"]*second_order_accel_loss(seq)
    if w.get("div_smooth",0)>0: loss += w["div_smooth"]*divergence_like_smoothness(seq)
    if w.get("boundary",0)>0: loss += w["boundary"]*boundary_penalty(seq)
    return loss
