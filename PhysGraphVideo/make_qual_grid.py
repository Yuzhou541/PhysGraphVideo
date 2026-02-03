
#!/usr/bin/env python3
# Build a qualitative grid PDF of video frames: rows=methods, cols=frames (evenly sampled).
# Inputs: directories for each method containing videos with identical stems (*.mp4/*.gif/*.webm).
# Usage:
#   python make_qual_grid.py --methods ours:runs/ours  dit:runs/dit_base  ctrl:runs/control  ldm:runs/ldm_video \
#       --out figs/qual_grid.pdf --n_frames 6 --resize 360 --pick 12 --seed 0
#
# Optional: provide an explicit list of stems with --stems file.txt (one stem per line, no extension).
#
import argparse, os, glob, math, random, re
from pathlib import Path
from typing import List, Dict
import numpy as np
import cv2
import matplotlib.pyplot as plt

def list_videos(d):
    vids = []
    for ext in ("*.mp4", "*.webm", "*.avi", "*.mov", "*.mkv", "*.gif"):
        vids.extend(sorted(Path(d).glob(ext)))
    return vids

def common_stems(method_dirs: Dict[str, str]) -> List[str]:
    sets = []
    for m, d in method_dirs.items():
        stems = {p.stem for p in list_videos(d)}
        sets.append(stems)
    common = set.intersection(*sets) if sets else set()
    return sorted(list(common))

def sample_frames(video_path: Path, n_frames: int, resize_short: int = None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(frames-1, 0), n_frames, dtype=int)
    imgs = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize_short is not None and min(frame.shape[:2]) > 0:
            h, w = frame.shape[:2]
            if h < w:
                new_h = resize_short
                new_w = int(w * new_h / h)
            else:
                new_w = resize_short
                new_h = int(h * new_w / w)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        imgs.append(frame)
    cap.release()
    return imgs

def draw_grid(method_dirs: Dict[str, str], stems: List[str], n_frames: int, resize_short: int, out_path: str, seed: int):
    random.seed(seed)
    if stems:
        stems = stems
    else:
        stems = common_stems(method_dirs)
    if not stems:
        raise RuntimeError("No common stems found across methods. Name your videos consistently.")
    # pick subset if too many
    pick = min(len(stems), args.pick) if args.pick else len(stems)
    stems = stems[:pick]

    methods = list(method_dirs.keys())
    n_methods = len(methods)
    cols = n_frames
    rows = n_methods * len(stems)

    # Figure size heuristic: width 10in, height per row 1.2in
    fig_w = 10
    fig_h = max(3.0, 1.2 * rows + 0.6)  # avoid too tall figure
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    if cols == 1:
        axes = np.expand_dims(axes, 1)

    # iterate
    row_idx = 0
    for stem in stems:
        # header row (text left)
        for m_i, m in enumerate(methods):
            vpath = None
            for ext in ("mp4","webm","avi","mov","mkv","gif"):
                cand = Path(method_dirs[m]) / f"{stem}.{ext}"
                if cand.exists():
                    vpath = cand; break
            if vpath is None:
                raise RuntimeError(f"Missing video for method={m}, stem={stem}")
            imgs = sample_frames(vpath, n_frames, resize_short)
            for c, img in enumerate(imgs):
                ax = axes[row_idx + m_i, c]
                ax.imshow(img)
                ax.set_axis_off()
                if c == 0:
                    ax.text(0.01, 0.98, f"{m}", fontsize=9, fontweight="bold",
                            ha="left", va="top", transform=ax.transAxes,
                            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2.5))
            # add stem title above first method of this block
        # add stem label above the group
        for c in range(cols):
            ax = axes[row_idx, c]
            ax.text(0.5, 1.02, stem, fontsize=9, ha="center", va="bottom", transform=ax.transAxes)
        row_idx += n_methods

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout(pad=0.1)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", required=True, help="List like name:dir ...")
    parser.add_argument("--stems", type=str, default=None, help="Optional text file listing stems to include")
    parser.add_argument("--n_frames", type=int, default=6)
    parser.add_argument("--resize", type=int, default=360, help="short side resize (px)")
    parser.add_argument("--out", type=str, default="figs/qual_grid.pdf")
    parser.add_argument("--pick", type=int, default=6, help="max samples to include")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    method_dirs = {}
    for item in args.methods:
        if ":" not in item:
            raise SystemExit("Each --methods entry must be name:dir")
        name, d = item.split(":", 1)
        method_dirs[name] = d

    stems = None
    if args.stems:
        with open(args.stems, "r", encoding="utf-8") as f:
            stems = [ln.strip() for ln in f if ln.strip()]

    draw_grid(method_dirs, stems, args.n_frames, args.resize, args.out, args.seed)
