# -*- coding: utf-8 -*-
# fig_tools/make_plan_vs_real_from_video_cv2.py
# 只用 OpenCV+NumPy，从视频本身估计 realized 轨迹（帧间运动质心），
# planned 轨迹用平滑+抽点；在等距帧上叠加虚/实线路径，输出拼图 PNG。
# 用法（PowerShell）：
#   python fig_tools\make_plan_vs_real_from_video_cv2.py ^
#     --video_dir data\clevrer\videos\val ^
#     --try_top 50 --n_frames 6 --short 360 ^
#     --out figs\plan_vs_real_oracle.png
import argparse, os
from pathlib import Path
import numpy as np
import cv2

EXTS = ("mp4","webm","avi","mov","mkv","gif")

def list_videos(video_dir: Path):
    vids = []
    for ext in EXTS:
        vids += sorted(video_dir.glob(f"*.{ext}"))
    return vids

def sample_frames_and_centroids(video_path: Path, n_frames=6, short_side=360):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 1:
        cap.release()
        return None  # too short

    # 先一遍扫全视频，估每帧“运动质心”（或亮度质心） —— realized 轨迹
    realized = []
    frames_gray = []
    prev = None
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for _ in range(total):
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        frames_gray.append(g)
        if prev is None:
            prev = g
            # 用亮度分布估一个初始质心
            c = weighted_centroid(g)
            realized.append(c)
            continue
        diff = cv2.absdiff(g, prev)
        prev = g
        c = motion_centroid(diff, fallback_img=g)
        realized.append(c)

    cap.release()

    # 过滤：至少要有两个不同点
    rc = np.array(realized, dtype=float)  # (T,2)
    if rc.shape[0] < 2 or np.allclose(rc[0], rc[-1]):
        return None

    # 计划轨迹：平滑+抽点
    planned = smooth_and_downsample(rc, points=8, k=7)

    # 再次打开，按等距索引采样帧图像（缩放短边），并叠加轨迹
    cap = cv2.VideoCapture(str(video_path))
    idxs = np.linspace(0, total-1, n_frames, dtype=int)

    panels = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if not ok: break
        h, w = fr.shape[:2]
        # 等比例把短边缩到 short_side
        if h < w:
            nh = short_side; nw = int(w * nh / h)
        else:
            nw = short_side; nh = int(h * nw / w)
        scale_x = nw / w
        scale_y = nh / h
        fr = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_AREA)

        # 画 planned（全局）：虚线（蓝）
        planned_scaled = planned * np.array([scale_x, scale_y])[None, :]
        draw_dashed_polyline(fr, planned_scaled, color=(255, 180, 80), thickness=2, dash=12, gap=8)

        # 画 realized（到当前帧）：实线（橙）
        r_now = rc[: int(idx)+1] * np.array([scale_x, scale_y])[None, :]
        if r_now.shape[0] >= 2:
            pts = r_now.astype(int).reshape(-1,1,2)
            cv2.polylines(fr, [pts], isClosed=False, color=(30,160,240), thickness=2, lineType=cv2.LINE_AA)

        # 角标
        txt = f"{video_path.stem} | t={int(idx)}"
        put_label(fr, txt)
        panels.append(fr)

    cap.release()
    if not panels:
        return None

    # 横向拼接
    hmin = min(p.shape[0] for p in panels)
    panels = [cv2.resize(p, (int(p.shape[1]*hmin/p.shape[0]), hmin), interpolation=cv2.INTER_AREA) for p in panels]
    grid = cv2.hconcat(panels)
    return grid

def motion_centroid(diff_gray, fallback_img=None):
    # 对差分图做高斯模糊，阈值取分位数，算加权质心
    g = cv2.GaussianBlur(diff_gray, (0,0), 3)
    thr = np.percentile(g, 98)
    mask = (g > max(10, thr*0.6)).astype(np.uint8)
    if mask.sum() < 50 and fallback_img is not None:
        # 动作太弱，退化到亮度质心
        return weighted_centroid(fallback_img)
    return weighted_centroid(g, mask=mask)

def weighted_centroid(gray, mask=None):
    gray = gray.astype(np.float32)
    if mask is None:
        # 亮度分布权重
        w = gray
    else:
        w = (gray * (mask>0)).astype(np.float32)
    s = w.sum()
    h, wi = w.shape
    if s < 1e-5:
        return np.array([wi/2.0, h/2.0], dtype=float)
    ys, xs = np.mgrid[0:h, 0:wi]
    cx = (w * xs).sum() / s
    cy = (w * ys).sum() / s
    return np.array([cx, cy], dtype=float)

def smooth_and_downsample(arr_xy, points=8, k=7):
    # arr_xy: (T,2)
    k = max(1, int(k)); k = k if k % 2 == 1 else k+1
    def movavg(a, k):
        if len(a) < k: return a.copy()
        pad = k//2
        ap = np.pad(a, (pad,pad), mode="edge")
        ker = np.ones(k)/k
        return np.convolve(ap, ker, mode="valid")
    xs = movavg(arr_xy[:,0], k)
    ys = movavg(arr_xy[:,1], k)
    n = min(max(6, points), 32)
    idx = np.linspace(0, len(xs)-1, num=n, dtype=int)
    return np.stack([xs[idx], ys[idx]], axis=1)

def put_label(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6; thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 6
    x0, y0 = 6, 6
    cv2.rectangle(img, (x0, y0), (x0+tw+2*pad, y0+th+2*pad), (255,255,255), -1)
    cv2.putText(img, text, (x0+pad, y0+pad+th-3), font, scale, (0,0,0), thick, cv2.LINE_AA)

def draw_dashed_polyline(img, pts, color, thickness=2, dash=12, gap=8):
    pts = np.asarray(pts, dtype=float)
    for i in range(len(pts)-1):
        p1 = pts[i]; p2 = pts[i+1]
        seg = p2 - p1
        L = float(np.linalg.norm(seg))
        if L < 1e-3: continue
        v = seg / L
        s = 0.0
        while s < L:
            e = min(s + dash, L)
            a = (p1 + v*s).astype(int)
            b = (p1 + v*e).astype(int)
            cv2.line(img, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)
            s += dash + gap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--try_top", type=int, default=50, help="最多尝试前 N 个视频直至成功")
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--short", type=int, default=360)
    ap.add_argument("--out", type=str, default="figs/plan_vs_real_oracle.png")
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    os.makedirs(Path(args.out).parent, exist_ok=True)

    vids = list_videos(video_dir)[:max(1,args.try_top)]
    last_err = None
    for vp in vids:
        try:
            grid = sample_frames_and_centroids(vp, n_frames=args.n_frames, short_side=args.short)
            if grid is None:
                print(f"[skip] {vp.name}: insufficient motion or frames")
                continue
            cv2.imwrite(args.out, grid)
            print(f"[OK] saved {args.out} using video={vp.name}")
            return
        except Exception as e:
            print(f"[skip] {vp.name}: {e}")
            last_err = str(e)

    raise RuntimeError(f"All {len(vids)} candidates failed. Last: {last_err}")

if __name__ == "__main__":
    main()
