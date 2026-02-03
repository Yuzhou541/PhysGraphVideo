# -*- coding: utf-8 -*-
import argparse, os
from types import SimpleNamespace as NS

import torch
from tqdm import tqdm

from physgraph.data.datasets import make_dataloader
from physgraph.models.generator import VideoGenerator
from physgraph.utils.metrics import video_psnr, video_ssim

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
    try: return int(x)
    except Exception: return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    dev = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    data  = getattr(cfg, "data", NS())
    train = getattr(cfg, "train", NS())
    evalc = getattr(cfg, "eval", NS())
    model_cfg = getattr(cfg, "model", NS())

    name      = getattr(data, "name", "bair")
    root      = getattr(data, "root", "data/bair")
    split_ev  = getattr(evalc, "split", getattr(data, "split", "val"))

    size      = _as_int(getattr(data, "image_size", getattr(data, "size", 64)), 64)
    channels  = _as_int(getattr(data, "channels", 3), 3)
    T_in      = _as_int(getattr(data, "T_in", getattr(data, "obs_len", 16)), 16)
    T_out     = _as_int(getattr(data, "T_out", getattr(data, "pred_len", 16)), 16)
    stride    = _as_int(getattr(data, "stride", 1), 1)

    batch     = _as_int(getattr(evalc, "batch", getattr(train, "batch", getattr(train, "batch_size", 8))), 8)
    workers   = _as_int(getattr(evalc, "num_workers", getattr(train, "num_workers", 4)), 4)
    seed      = _as_int(getattr(train, "seed", 0), 0)

    # dataloader
    loader = make_dataloader(
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


    # model
    in_ch  = _as_int(getattr(model_cfg, "in_ch", channels), channels)
    out_ch = _as_int(getattr(model_cfg, "out_ch", channels), channels)
    lat_ch = _as_int(getattr(model_cfg, "lat_ch", 128), 128)
    model = VideoGenerator(in_ch=in_ch, out_ch=out_ch, lat_ch=lat_ch).to(dev)

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    sd = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(sd, strict=False)
    model.eval()

    psnrs, ssims, n = [], [], 0
    with torch.no_grad():
        for (inp, tgt) in tqdm(loader, desc=f"[Eval:{split_ev}]", ncols=90):
            inp_TB = inp.permute(1, 0, 2, 3, 4).to(dev, non_blocking=True)
            tgt_TB = tgt.permute(1, 0, 2, 3, 4).to(dev, non_blocking=True)

            seq_TB = torch.cat([inp_TB, tgt_TB], dim=0)
            pred_TB = model(seq_TB, cond=None)
            if pred_TB.shape[0] == seq_TB.shape[0]:
                pred_out = pred_TB[T_in:]
            else:
                pred_out = pred_TB

            v_pred = pred_out.permute(1, 0, 2, 3, 4).contiguous()
            v_tgt  = tgt_TB.permute(1, 0, 2, 3, 4).contiguous()

            psnrs.append(float(video_psnr(v_pred, v_tgt)))
            ssims.append(float(video_ssim(v_pred, v_tgt)))
            n += v_pred.shape[0]

    out = {
        "split": split_ev,
        "samples": n,
        "psnr": sum(psnrs) / max(1, len(psnrs)),
        "ssim": sum(ssims) / max(1, len(ssims)),
    }
    print(out)

if __name__ == "__main__":
    main()
