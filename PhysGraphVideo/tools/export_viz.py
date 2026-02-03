# File: tools/export_viz.py
# Robust visualizer for predicted video sequences.
# - Auto-finds the correct checkpoint if the provided path is missing.
# - Matches the train/eval pipeline: layout normalization, RAG conditioning, future-frames viz.
# - Writes a GIF to --out_gif.
#
# Usage (VS Code terminal, Windows):
#   python -m tools.export_viz --config configs\phys_plus_rag.yaml --device cuda --out_gif runs\sample.gif
#   # (optional) explicitly pass a ckpt:
#   python -m tools.export_viz --config configs\phys_plus_rag.yaml --ckpt runs\phys_plus_rag\best.pt --device cuda --out_gif runs\sample.gif

from __future__ import annotations
import argparse
import os
import json
import glob
from typing import List, Tuple, Optional

import torch
import imageio.v2 as imageio

from physgraph.utils.common import load_config, to_ns
from physgraph.data.datasets import make_dataloader
from physgraph.models.generator import VideoGenerator
from physgraph.models.text_encoder import TextEncoder
from physgraph.rag.prompts import to_tokens


# -------------------- small utils (consistent with train/evaluate) --------------------
def _as_int(x):
    if isinstance(x, int): return x
    if isinstance(x, float): return int(x)
    try: return int(float(str(x).strip()))
    except Exception: raise ValueError(f"Expected int-like, got {x} ({type(x)})")

def _as_float(x):
    if isinstance(x, (int, float)): return float(x)
    try: return float(str(x).strip())
    except Exception: raise ValueError(f"Expected float-like, got {x} ({type(x)})")

def _as_bool(x):
    if isinstance(x, bool): return x
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def _cfg_get(ns, dotted: str, default=None):
    cur = ns
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur

def _to_TBCHW(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """
    Return tensor as (T,B,C,H,W). Detects (B,T,C,H,W) and permutes when needed.
    Returns (tensor_TB, was_BT)
    """
    if x.dim() != 5:
        raise ValueError(f"Expected 5D tensor, got {x.shape}")
    T_like, B_like = x.size(0), x.size(1)
    if B_like <= 1024 and T_like > 64 and B_like <= 64:
        return x, False  # already (T,B,...)
    if T_like <= 64 and B_like <= 256:
        return x.permute(1, 0, 2, 3, 4).contiguous(), True  # (B,T,...) -> (T,B,...)
    return x.permute(1, 0, 2, 3, 4).contiguous(), True

def _cat_time(inp: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Accept inp,tgt in either layout; return:
      seq_TB: concatenated along time in (T,B,C,H,W)
      tgt_TB: target in (T,B,C,H,W)
      T_in:   input time length
    """
    # If shapes are (B,T,...) and (B,T',...) -> cat along dim=1 then permute.
    if inp.shape[:1] == tgt.shape[:1] and inp.shape[1] != tgt.shape[1]:
        T_in = inp.shape[1]
        seq_BT = torch.cat([inp, tgt], dim=1)
        seq_TB = seq_BT.permute(1, 0, 2, 3, 4).contiguous()
        tgt_TB = tgt.permute(1, 0, 2, 3, 4).contiguous()
        return seq_TB, tgt_TB, T_in

    # Otherwise normalize both to (T,B,...) then cat along time.
    inp_TB, _ = _to_TBCHW(inp)
    tgt_TB, _ = _to_TBCHW(tgt)
    T_in = inp_TB.shape[0]
    seq_TB = torch.cat([inp_TB, tgt_TB], dim=0)
    return seq_TB, tgt_TB, T_in

def _setup_rag(cfg):
    enabled = _as_bool(_cfg_get(cfg, "rag.enabled", False))
    if not enabled:
        return [""]
    # Import lazily to avoid heavy deps when disabled
    from physgraph.rag.graph_builder import build_sample_graph, export_jsonl
    from physgraph.rag.retriever import GraphRAG

    os.makedirs("runs/rag", exist_ok=True)
    jsonl = "runs/rag/sample_kg.jsonl"
    export_jsonl(build_sample_graph(), jsonl)
    topk = _as_int(_cfg_get(cfg, "rag.topk", 4))
    query = str(_cfg_get(cfg, "rag.query", "physics priors for video dynamics"))
    rag = GraphRAG(jsonl, topk=topk)
    return rag.query(query)


# -------------------- checkpoint discovery --------------------
def _hint_from_config_path(cfg_path: str) -> str:
    base = os.path.basename(cfg_path)
    name, _ = os.path.splitext(base)
    return name  # e.g., "phys_plus_rag"

def _read_manifest_candidates() -> List[str]:
    cands = []
    for manifest in ("runs/ablate/ckpts.json", "runs/ablate/manifest.json"):
        if os.path.isfile(manifest):
            try:
                with open(manifest, "r", encoding="utf-8") as f:
                    m = json.load(f)
                for k, v in m.items():
                    if isinstance(v, str):
                        cands.append(v)
            except Exception:
                pass
    return cands

def _find_ckpt(args_ckpt: Optional[str], cfg, exp_hint: str) -> str:
    tried = []

    def _exists(p: str) -> Optional[str]:
        if p and os.path.isfile(p):
            return os.path.normpath(p)
        tried.append(os.path.normpath(p))
        return None

    # 1) user-provided path
    if args_ckpt:
        p = _exists(args_ckpt)
        if p: return p

    # 2) config-specified ckpt_dir
    ckpt_dir = str(_cfg_get(cfg, "train.ckpt_dir", "runs/exp"))
    p = _exists(os.path.join(ckpt_dir, "best.pt"))
    if p: return p

    # 3) exp_hint under runs/
    p = _exists(os.path.join("runs", exp_hint, "best.pt"))
    if p: return p

    # 4) common directories used in this repo
    for d in ("cvpr_base", "ablation_physics", "ablation_graphrag", "phys_plus_rag", "exp"):
        p = _exists(os.path.join("runs", d, "best.pt"))
        if p: return p

    # 5) manifest from one_click_eval (if any)
    for cand in _read_manifest_candidates():
        p = _exists(cand)
        if p: return p

    # 6) last resort: search for recent best.pt under runs/
    best, best_mtime = None, -1.0
    for cand in glob.glob(os.path.join("runs", "**", "best.pt"), recursive=True):
        try:
            m = os.path.getmtime(cand)
        except OSError:
            continue
        if m > best_mtime:
            best, best_mtime = cand, m
    if best:
        return os.path.normpath(best)

    # Fail with helpful message
    msg = ["Could not locate checkpoint. Tried:"]
    msg += [f"  - {p}" for p in tried]
    raise FileNotFoundError("\n".join(msg))


# -------------------- visualization helpers --------------------
def _to_uint8_frames(frames_TCHW: torch.Tensor) -> List:
    """
    frames_TCHW: [T, C, H, W], values in [0,1] (float)
    returns: list of HxW or HxWx3 uint8 numpy arrays
    """
    frames_TCHW = frames_TCHW.detach().clamp(0, 1).cpu()
    T, C, H, W = frames_TCHW.shape
    out = []
    for t in range(T):
        x = frames_TCHW[t]
        if C == 1:
            img = (x[0] * 255.0).round().numpy().astype("uint8")
        else:
            img = (x.permute(1, 2, 0) * 255.0).round().numpy().astype("uint8")
        out.append(img)
    return out


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="Optional checkpoint path; if missing, auto-discover.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_gif", required=True, help="Output GIF path, e.g., runs/sample.gif")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    cfg = to_ns(load_config(args.config))
    device = args.device
    exp_hint = _hint_from_config_path(args.config)

    # ---- Data params (match evaluate.py defaults) ----
    seq_len       = _as_int(_cfg_get(cfg, "data.seq_len", 16))
    pred_len      = _as_int(_cfg_get(cfg, "data.pred_len", 12))
    image_size    = _as_int(_cfg_get(cfg, "data.image_size", 64))
    batch_size    = _as_int(_cfg_get(cfg, "data.batch_size", 8))
    dataset_name  = str(_cfg_get(cfg, "data.name", "bouncing_balls"))
    num_sequences = _as_int(_cfg_get(cfg, f"data.{args.split}_sequences", 4))

    # ---- Build loader but only need a small sample ----
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
)

    # ---- Model ----
    in_channels = _as_int(_cfg_get(cfg, "model.in_channels", 1))
    hidden_dim  = _as_int(_cfg_get(cfg, "model.hidden_dim", 96))
    film_dim    = _as_int(_cfg_get(cfg, "model.film_dim", 128))
    num_layers  = _as_int(_cfg_get(cfg, "model.num_layers", 2))

    model = VideoGenerator(
        in_channels=in_channels, hidden_dim=hidden_dim, film_dim=film_dim, num_layers=num_layers
    ).to(device)
    text_enc = TextEncoder(film_dim=film_dim).to(device)

    # ---- Find & load checkpoint robustly ----
    ckpt_path = _find_ckpt(args.ckpt, cfg, exp_hint)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval(); text_enc.eval()

    # ---- RAG ----
    rag_texts = _setup_rag(cfg)

    # ---- Run one mini-batch and visualize the FIRST sample in the batch ----
    os.makedirs(os.path.dirname(args.out_gif), exist_ok=True)
    frames = []  # will hold uint8 frames for GIF

    with torch.no_grad():
        for (inp, tgt) in loader:
            inp = inp.to(device)
            tgt = tgt.to(device)

            seq_TB, tgt_TB, T_in = _cat_time(inp, tgt)
            B = seq_TB.size(1)

            tokens = torch.tensor(to_tokens(["; ".join(rag_texts)] * B), device=device)
            cond = text_enc(tokens)

            pred_TB = model(seq_TB, cond)              # [T_in+T_out, B, C, H, W]
            pred_f  = pred_TB[T_in:]                    # future frames [T_out, B, C, H, W]

            # Take first sample in batch
            pred_one = pred_f[:, 0]                     # [T_out, C, H, W]
            frames = _to_uint8_frames(pred_one)
            break  # visualize only one batch

    if not frames:
        raise RuntimeError("No frames produced for visualization.")

    imageio.mimsave(args.out_gif, frames, fps=10)
    print({"gif": os.path.normpath(args.out_gif), "ckpt_used": os.path.normpath(ckpt_path)})


if __name__ == "__main__":
    main()
