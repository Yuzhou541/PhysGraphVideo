# -*- coding: utf-8 -*-
# fig_tools/make_qual_grid_cv2.py
# 只用 OpenCV+NumPy 生成数据集预览网格（无 Matplotlib）
# 用法:
#   python fig_tools\make_qual_grid_cv2.py --video_dir data\clevrer\videos\val ^
#       --stems lists\qual_list.txt --n_frames 6 --short 360 --out figs\qual_grid_ref.png
import argparse, os
from pathlib import Path
import numpy as np
import cv2

EXTS = ("mp4","webm","avi","mov","mkv","gif")

def find_video(video_dir: Path, stem: str):
    for ext in EXTS:
        p = video_dir / f"{stem}.{ext}"
        if p.exists(): return p
    # 兼容加前缀 video_
    for ext in EXTS:
        p = video_dir / f"video_{stem}.{ext}"
        if p.exists(): return p
    # 兼容有数字 id 的情况：取数字后匹配
    digits = "".join([c for c in stem if c.isdigit()])
    if digits:
        for ext in EXTS:
            p = video_dir / f"{digits}.{ext}"
            if p.exists(): return p
            p = video_dir / f"video_{digits}.{ext}"
            if p.exists(): return p
    return None

def sample_frames(video_path: Path, n=6, short_side=360):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total-1, 0), n, dtype=int)
    imgs = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok: break
        h, w = fr.shape[:2]
        # 等比例把短边缩放到 short_side
        if h < w:
            nh = short_side
            nw = int(w * nh / h)
        else:
            nw = short_side
            nh = int(h * nw / w)
        fr = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_AREA)
        imgs.append(fr)
    cap.release()
    return imgs

def put_label(img, text):
    # 左上角贴白底文字
    overlay = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    th = 1
    (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
    pad = 6
    cv2.rectangle(overlay, (6,6), (6+tw+2*pad, 6+tht+2*pad), (255,255,255), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.putText(img, text, (6+pad, 6+pad+tht-3), font, scale, (0,0,0), th, cv2.LINE_AA)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--stems", type=str, default=None, help="每行一个不带扩展名的文件名")
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--short", type=int, default=360, help="短边缩放")
    ap.add_argument("--out", type=str, default="figs/qual_grid_ref.png")
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 选择样例 stems
    if args.stems and Path(args.stems).exists():
        stems = [ln.strip() for ln in Path(args.stems).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        stems = [p.stem for p in video_dir.glob("*.mp4")][:6]

    rows = []
    for stem in stems:
        vp = find_video(video_dir, stem)
        if vp is None:
            raise RuntimeError(f"Video not found for stem: {stem}")
        imgs = sample_frames(vp, args.n_frames, args.short)
        if not imgs: continue
        # 给第一帧贴上 stem 标签
        imgs[0] = put_label(imgs[0], stem)
        # 统一高度，用 hconcat
        hmin = min(im.shape[0] for im in imgs)
        imgs = [cv2.resize(im, (int(im.shape[1]*hmin/im.shape[0]), hmin), interpolation=cv2.INTER_AREA) for im in imgs]
        row = cv2.hconcat(imgs)
        rows.append(row)

    # 统一宽度，用 vconcat（补齐到相同宽度）
    maxw = max(r.shape[1] for r in rows)
    pad_rows = []
    for r in rows:
        if r.shape[1] < maxw:
            pad = np.zeros((r.shape[0], maxw - r.shape[1], 3), dtype=r.dtype)
            r = np.concatenate([r, pad], axis=1)
        pad_rows.append(r)
    grid = cv2.vconcat(pad_rows)
    cv2.imwrite(args.out, grid)
    print(f"[OK] saved {args.out}")

if __name__ == "__main__":
    main()
