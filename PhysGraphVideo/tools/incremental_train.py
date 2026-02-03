# File: tools/incremental_train.py
# Incremental fine-tuning with robust time concat and optional KD/Replay.
# Usage (VSCode terminal, Windows):
#   python -m tools.incremental_train --base_ckpt runs\cvpr_base\best.pt --config configs\incremental_phase2.yaml --device cuda

from __future__ import annotations
import argparse
import os
import math
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from physgraph.utils.common import load_config, to_ns
from physgraph.utils.metrics import video_psnr
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
    # If looks like (B,T,...)
    if x.size(0) <= 256 and x.size(1) <= 128:
        return x.permute(1, 0, 2, 3, 4).contiguous(), True
    return x, False

def _cat_time(inp: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Accept inp,tgt in either layout; return:
      seq_TB: concatenated along time in (T,B,C,H,W)
      tgt_TB: target in (T,B,C,H,W)
      T_in:   input time length
    Supports different lengths (e.g., T_in=16, T_out=12).
    """
    # (B,T,...) safe path: cat on dim=1 then permute
    if inp.dim() == 5 and tgt.dim() == 5 and inp.shape[0] == tgt.shape[0] and inp.shape[1] != tgt.shape[1]:
        T_in = inp.shape[1]
        seq_BT = torch.cat([inp, tgt], dim=1)
        seq_TB = seq_BT.permute(1, 0, 2, 3, 4).contiguous()
        tgt_TB = tgt.permute(1, 0, 2, 3, 4).contiguous()
        return seq_TB, tgt_TB, T_in

    # Normalize双方到 (T,B,...) 再拼接
    inp_TB, _ = _to_TBCHW(inp)
    tgt_TB, _ = _to_TBCHW(tgt)
    T_in = inp_TB.shape[0]
    seq_TB = torch.cat([inp_TB, tgt_TB], dim=0)
    return seq_TB, tgt_TB, T_in


# -------------------- RAG setup (optional) --------------------
def _setup_rag(cfg):
    enabled = _as_bool(_cfg_get(cfg, "rag.enabled", False))
    if not enabled:
        return [""]
    from physgraph.rag.graph_builder import build_sample_graph, export_jsonl
    from physgraph.rag.retriever import GraphRAG

    os.makedirs("runs/rag", exist_ok=True)
    jsonl = "runs/rag/inc_kg.jsonl"
    export_jsonl(build_sample_graph(), jsonl)
    topk = _as_int(_cfg_get(cfg, "rag.topk", 4))
    query = str(_cfg_get(cfg, "rag.query", "phase-2 physics priors"))
    rag = GraphRAG(jsonl, topk=topk)
    return rag.query(query)


# -------------------- KD teacher (optional) --------------------
class _FrozenTeacher(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        import copy
        self.teacher = copy.deepcopy(model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, seq_TB: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.teacher(seq_TB, cond)


# -------------------- incremental train --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True, help="Path to base checkpoint to initialize from.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = to_ns(load_config(args.config))
    device = args.device

    # ---- data params ----
    seq_len       = _as_int(_cfg_get(cfg, "data.seq_len", 16))
    pred_len      = _as_int(_cfg_get(cfg, "data.pred_len", 12))
    image_size    = _as_int(_cfg_get(cfg, "data.image_size", 64))
    batch_size    = _as_int(_cfg_get(cfg, "data.batch_size", 8))
    dataset_name  = str(_cfg_get(cfg, "data.name", "bouncing_balls"))
    num_sequences = _as_int(_cfg_get(cfg, "data.train_sequences", 1000))

    # optional replay
    replay_name      = str(_cfg_get(cfg, "data.replay_name", ""))
    replay_sequences = _as_int(_cfg_get(cfg, "data.replay_sequences", 0))

    # ---- model params ----
    in_channels = _as_int(_cfg_get(cfg, "model.in_channels", 1))
    hidden_dim  = _as_int(_cfg_get(cfg, "model.hidden_dim", 96))
    film_dim    = _as_int(_cfg_get(cfg, "model.film_dim", 128))
    num_layers  = _as_int(_cfg_get(cfg, "model.num_layers", 2))

    # ---- train params ----
    epochs        = _as_int(_cfg_get(cfg, "train.epochs", 5))
    lr            = _as_float(_cfg_get(cfg, "train.lr", 2e-4))
    weight_decay  = _as_float(_cfg_get(cfg, "train.weight_decay", 1e-4))
    recon_w       = _as_float(_cfg_get(cfg, "loss.recon", 1.0))
    kd_w          = _as_float(_cfg_get(cfg, "inc.kd_w", 0.0))        # KD to base
    l2anchor_w    = _as_float(_cfg_get(cfg, "inc.l2_anchor", 0.0))   # L2 to base weights

    # ---- loaders ----
    train_loader = make_dataloader(
        dataset_name, "train", batch_size,
        seq_len=seq_len, pred_len=pred_len, size=image_size, num_sequences=num_sequences, shuffle=True
    )
    val_loader = make_dataloader(
        dataset_name, "val", batch_size=8,
        seq_len=seq_len, pred_len=pred_len, size=image_size, num_sequences=_as_int(_cfg_get(cfg, "data.val_sequences", 4))
    )

    replay_loader = None
    if replay_name and replay_sequences > 0:
        replay_loader = make_dataloader(
            replay_name, "train", batch_size,
            seq_len=seq_len, pred_len=pred_len, size=image_size, num_sequences=replay_sequences, shuffle=True
        )

    # ---- model / text encoder ----
    model = VideoGenerator(
        in_channels=in_channels, hidden_dim=hidden_dim, film_dim=film_dim, num_layers=num_layers
    ).to(device)
    text_enc = TextEncoder(film_dim=film_dim).to(device)

    # Load base checkpoint into student model
    model.load_state_dict(torch.load(args.base_ckpt, map_location=device))

    # Optional KD teacher snapshot from base
    teacher = _FrozenTeacher(model).to(device) if kd_w > 0 else None

    # ---- optimizer ----
    opt = optim.AdamW(list(model.parameters()) + list(text_enc.parameters()), lr=lr, weight_decay=weight_decay)

    # L2-anchor target (flattened base params) if needed
    base_params_vec = None
    if l2anchor_w > 0:
        with torch.no_grad():
            base_params_vec = torch.cat([p.detach().flatten() for p in model.parameters()]).to(device)

    # ---- RAG texts ----
    rag_texts = _setup_rag(cfg)

    # ---- run dir ----
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    run_dir = os.path.join("runs", cfg_name)
    os.makedirs(run_dir, exist_ok=True)
    best_psnr, best_path = -1.0, os.path.join(run_dir, "best.pt")

    # -------------------- training --------------------
    for ep in range(1, epochs + 1):
        model.train(); text_enc.train()
        pbar = tqdm(train_loader, desc=f"[IncTrain] {ep}/{epochs}", ncols=100)

        for (inp, tgt) in pbar:
            inp = inp.to(device)  # [B,T_in,C,H,W] or [T_in,B,C,H,W]
            tgt = tgt.to(device)  # [B,T_out,C,H,W] or [T_out,B,C,H,W]

            # 主任务 batch
            seq_TB, tgt_TB, T_in = _cat_time(inp, tgt)   # [T_in+T_out,B,C,H,W], [T_out,B,C,H,W]
            B = seq_TB.size(1)

            # 条件编码（RAG 或占位）
            tokens = torch.tensor(to_tokens(["; ".join(rag_texts)] * B), device=device)
            cond = text_enc(tokens)

            # 预测未来帧
            pred_TB = model(seq_TB, cond)     # [T_in+T_out,B,C,H,W]
            pred_f  = pred_TB[T_in:]          # [T_out,B,C,H,W]

            # 重建损失
            loss = 0.0
            recon = F.l1_loss(pred_f, tgt_TB)
            loss = loss + recon_w * recon

            # KD 到基模型（teacher）
            if teacher is not None and kd_w > 0:
                with torch.no_grad():
                    teach_out = teacher(seq_TB, cond)[T_in:]
                kd = F.mse_loss(pred_f, teach_out)
                loss = loss + kd_w * kd

            # L2 to base (EWC 简化版)
            if l2anchor_w > 0 and base_params_vec is not None:
                cur_vec = torch.cat([p.flatten() for p in model.parameters()])
                l2_anchor = F.mse_loss(cur_vec, base_params_vec)
                loss = loss + l2anchor_w * l2_anchor

            # 可选 Replay 一个 batch（若配置了 replay 数据）
            if replay_loader is not None:
                try:
                    r_inp, r_tgt = next(_replay_iter)
                except NameError:
                    _replay_iter = iter(replay_loader)
                    r_inp, r_tgt = next(_replay_iter)
                except StopIteration:
                    _replay_iter = iter(replay_loader)
                    r_inp, r_tgt = next(_replay_iter)

                r_inp = r_inp.to(device); r_tgt = r_tgt.to(device)
                r_seq_TB, r_tgt_TB, r_T_in = _cat_time(r_inp, r_tgt)
                rB = r_seq_TB.size(1)
                r_tokens = torch.tensor(to_tokens(["; ".join(rag_texts)] * rB), device=device)
                r_cond = text_enc(r_tokens)
                r_pred_TB = model(r_seq_TB, r_cond)
                r_pred_f = r_pred_TB[r_T_in:]
                # replay loss（和主任务同权重，可在需要时扩展成单独系数）
                loss = loss + recon_w * F.l1_loss(r_pred_f, r_tgt_TB)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        # ---- 验证 ----
        model.eval(); text_enc.eval()
        with torch.no_grad():
            psnrs: List[float] = []
            for (vinp, vtgt) in val_loader:
                vinp = vinp.to(device); vtgt = vtgt.to(device)
                v_seq_TB, v_tgt_TB, v_T_in = _cat_time(vinp, vtgt)
                Bv = v_seq_TB.size(1)
                v_tokens = torch.tensor(to_tokens(["; ".join(rag_texts)] * Bv), device=device)
                v_cond = text_enc(v_tokens)
                v_pred_TB = model(v_seq_TB, v_cond)
                v_pred_f  = v_pred_TB[v_T_in:]
                psnrs.append(video_psnr(v_pred_f, v_tgt_TB).item())
            cur_psnr = sum(psnrs) / max(1, len(psnrs))

        if cur_psnr > best_psnr:
            best_psnr = cur_psnr
            torch.save(model.state_dict(), best_path)

    print({"best_psnr": best_psnr, "ckpt": os.path.normpath(best_path)})


if __name__ == "__main__":
    main()
