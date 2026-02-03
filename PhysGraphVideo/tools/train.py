# -*- coding: utf-8 -*-
import argparse, os, time, math, json
from types import SimpleNamespace as NS

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from physgraph.data.datasets import make_dataloader
from physgraph.models.generator import VideoGenerator
from physgraph.utils.metrics import video_psnr, video_ssim

# ---------- tiny utils ----------

def _as_ns(d):
    if isinstance(d, dict):
        return NS(**{k: _as_ns(v) for k, v in d.items()})
    return d

def _load_yaml(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _as_ns(cfg)

def _as_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def _as_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ---------- train loop ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    dev = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # ---- read config safely ----
    data = getattr(cfg, "data", NS())
    train = getattr(cfg, "train", NS())
    model_cfg = getattr(cfg, "model", NS())

    name      = getattr(data, "name", "bair")
    root      = getattr(data, "root", "data/bair")
    split_tr  = getattr(data, "split", "train")  # default
    size      = _as_int(getattr(data, "size", getattr(data, "image_size", 64)), 64)
    channels  = _as_int(getattr(data, "channels", 3), 3)
    T_in      = _as_int(getattr(data, "T_in", getattr(data, "obs_len", 16)), 16)
    T_out     = _as_int(getattr(data, "T_out", getattr(data, "pred_len", 16)), 16)
    stride    = _as_int(getattr(data, "stride", 1), 1)

    epochs     = _as_int(getattr(train, "epochs", 5), 5)
    batch      = _as_int(getattr(train, "batch", getattr(train, "batch_size", 8)), 8)
    workers    = _as_int(getattr(train, "num_workers", 4), 4)
    lr         = _as_float(getattr(train, "lr", 1e-4), 1e-4)
    wd         = _as_float(getattr(train, "weight_decay", 1e-4), 1e-4)
    seed       = _as_int(getattr(train, "seed", 0), 0)
    out_dir    = getattr(train, "out_dir", f"runs/{name}")
    _ensure_dir(out_dir)

    g = torch.Generator()
    g.manual_seed(seed)

    # ---- dataloaders: use keyword args to avoid positional mis-order ----
    train_loader =     train_loader =     train_loader = make_dataloader(
        name=cfg.data.name,
        root=cfg.data.root,
        split='train',
        T_in=cfg.data.t_in,
        T_out=cfg.data.t_out,
        size=cfg.data.image_size,
        channels=getattr(cfg.data, 'channels', 3),
        stride=getattr(cfg.data, 'stride', 1),
        batch_size=getattr(cfg.data, 'batch_size', 8),
        num_workers=getattr(cfg.data, 'num_workers', 0),
        pin_memory=getattr(cfg.data, 'pin_memory', False),
        is_train=True,
    ),
    stride=getattr(cfg.data, 'stride', 1),
    batch_size=getattr(cfg.data, 'batch_size', 8),
    num_workers=getattr(cfg.data, 'num_workers', 0),
    pin_memory=getattr(cfg.data, 'pin_memory', False),
    is_train=True,


    # validation split: try 'val', otherwise fallback to 'test' or 'train'
    split_val = "val"
    try:
        val_loader =         val_loader =         val_loader = make_dataloader(
            name=cfg.data.name,
            root=cfg.data.root,
            split='val',
            T_in=cfg.data.t_in,
            T_out=cfg.data.t_out,
            size=cfg.data.image_size,
            channels=getattr(cfg.data, 'channels', 3),
            stride=getattr(cfg.data, 'stride', 1),
            batch_size=getattr(cfg.data, 'batch_size', 1),
            num_workers=getattr(cfg.data, 'num_workers', 0),
            pin_memory=getattr(cfg.data, 'pin_memory', False),
            is_train=False,
        ),
    stride=getattr(cfg.data, 'stride', 1),
    batch_size=getattr(cfg.data, 'batch_size', 1),
    num_workers=getattr(cfg.data, 'num_workers', 0),
    pin_memory=getattr(cfg.data, 'pin_memory', False),
    is_train=False,

    except Exception:
        split_val = "test"
        try:
            val_loader =             val_loader =             val_loader = make_dataloader(
                name=cfg.data.name,
                root=cfg.data.root,
                split='val',
                T_in=cfg.data.t_in,
                T_out=cfg.data.t_out,
                size=cfg.data.image_size,
                channels=getattr(cfg.data, 'channels', 3),
                stride=getattr(cfg.data, 'stride', 1),
                batch_size=getattr(cfg.data, 'batch_size', 1),
                num_workers=getattr(cfg.data, 'num_workers', 0),
                pin_memory=getattr(cfg.data, 'pin_memory', False),
                is_train=False,
            ),
    stride=getattr(cfg.data, 'stride', 1),
    batch_size=getattr(cfg.data, 'batch_size', 1),
    num_workers=getattr(cfg.data, 'num_workers', 0),
    pin_memory=getattr(cfg.data, 'pin_memory', False),
    is_train=False,

        except Exception:
            # last fallback (small sanity val from train)
            val_loader =             val_loader =             val_loader = make_dataloader(
                name=cfg.data.name,
                root=cfg.data.root,
                split='val',
                T_in=cfg.data.t_in,
                T_out=cfg.data.t_out,
                size=cfg.data.image_size,
                channels=getattr(cfg.data, 'channels', 3),
                stride=getattr(cfg.data, 'stride', 1),
                batch_size=getattr(cfg.data, 'batch_size', 1),
                num_workers=getattr(cfg.data, 'num_workers', 0),
                pin_memory=getattr(cfg.data, 'pin_memory', False),
                is_train=False,
            ),
    stride=getattr(cfg.data, 'stride', 1),
    batch_size=getattr(cfg.data, 'batch_size', 1),
    num_workers=getattr(cfg.data, 'num_workers', 0),
    pin_memory=getattr(cfg.data, 'pin_memory', False),
    is_train=False,


    # ---- model / optim ----
    in_ch  = _as_int(getattr(model_cfg, "in_ch", channels), channels)
    out_ch = _as_int(getattr(model_cfg, "out_ch", channels), channels)
    lat_ch = _as_int(getattr(model_cfg, "lat_ch", 128), 128)

    model = VideoGenerator(in_ch=in_ch, out_ch=out_ch, lat_ch=lat_ch).to(dev)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss()

    best_psnr = -1.0
    best_ckpt = os.path.join(out_dir, "best.pt")

    # ---- training ----
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Train] {ep}/{epochs}", ncols=90)
        for (inp, tgt) in pbar:
            # inp/tgt: [B, T, C, H, W]  -> [T, B, C, H, W]
            inp_TB = inp.permute(1, 0, 2, 3, 4).to(dev, non_blocking=True)
            tgt_TB = tgt.permute(1, 0, 2, 3, 4).to(dev, non_blocking=True)
            seq_TB = torch.cat([inp_TB, tgt_TB], dim=0)  # [T_in+T_out,B,C,H,W]

            pred_TB = model(seq_TB, cond=None)  # model应返回 [T_in+T_out,B,C,H,W] 或至少包含预测段
            # 兼容两类实现：返回全时长或仅返回输出段
            if pred_TB.shape[0] == seq_TB.shape[0]:
                pred_out = pred_TB[T_in:]       # 取输出段
            else:
                pred_out = pred_TB

            loss = crit(pred_out, tgt_TB)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # ---- validation ----
        model.eval()
        psnrs = []
        with torch.no_grad():
            for (inp, tgt) in val_loader:
                inp_TB = inp.permute(1, 0, 2, 3, 4).to(dev, non_blocking=True)
                tgt_TB = tgt.permute(1, 0, 2, 3, 4).to(dev, non_blocking=True)
                seq_TB = torch.cat([inp_TB, tgt_TB], dim=0)

                pred_TB = model(seq_TB, cond=None)
                if pred_TB.shape[0] == seq_TB.shape[0]:
                    pred_out = pred_TB[T_in:]
                else:
                    pred_out = pred_TB

                # metrics expect [B,T,C,H,W]
                v_pred = pred_out.permute(1, 0, 2, 3, 4).contiguous()
                v_tgt  = tgt_TB.permute(1, 0, 2, 3, 4).contiguous()
                psnrs.append(float(video_psnr(v_pred, v_tgt)))

        cur = sum(psnrs) / max(1, len(psnrs))
        if cur > best_psnr:
            best_psnr = cur
            torch.save(model.state_dict(), best_ckpt)

    print({"best_psnr": best_psnr, "ckpt": best_ckpt})

if __name__ == "__main__":
    main()
