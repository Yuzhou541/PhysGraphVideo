# -*- coding: utf-8 -*-
# fig_tools/make_plan_vs_real_cv2.py  (robust)
# 只用 OpenCV+NumPy：等距帧叠加 planned(虚线, 平滑+抽点) 与 realized(实线, 选"最长轨迹"物体)。
# 找不到 CSV -> 用 annotation JSON 生成；若轨迹太短 -> 自动换下一个视频，直到成功。
# 用法示例:
#   python fig_tools\make_plan_vs_real_cv2.py ^
#     --video_dir data\clevrer\videos\val ^
#     --tracks_dir tracks ^
#     --ann_dir data\clevrer\annotations\val ^
#     --try_top 20 --n_frames 6 --short 360 ^
#     --out figs\plan_vs_real_oracle.png
import argparse, os, csv, json
from pathlib import Path
import numpy as np
import cv2

EXTS = ("mp4","webm","avi","mov","mkv","gif")

def digits_only(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())

def find_video(video_dir: Path, stem: str):
    for ext in EXTS:
        p = video_dir / f"{stem}.{ext}"
        if p.exists(): return p
    for ext in EXTS:
        p = video_dir / f"video_{stem}.{ext}"
        if p.exists(): return p
    digs = digits_only(stem)
    if digs:
        for ext in EXTS:
            p = video_dir / f"{digs}.{ext}"
            if p.exists(): return p
            p = video_dir / f"video_{digs}.{ext}"
            if p.exists(): return p
    return None

def match_track(tracks_dir: Path, stem: str):
    # 1) 完全同名
    cand = tracks_dir / f"{stem}.csv"
    if cand.exists(): return cand
    # 2) 加/去 video_ 前缀
    if stem.startswith("video_"):
        cand = tracks_dir / (stem[6:] + ".csv")
        if cand.exists(): return cand
    else:
        cand = tracks_dir / ("video_" + stem + ".csv")
        if cand.exists(): return cand
    # 3) 数字匹配
    digs = digits_only(stem)
    if digs:
        for name in (f"{digs}.csv", f"video_{digs}.csv"):
            cand = tracks_dir / name
            if cand.exists(): return cand
    # 4) 模糊匹配（包含关系）
    matches = sorted(tracks_dir.glob(f"{stem}*.csv"))
    if matches: return matches[0]
    return None

def get_bbox_center(obj: dict):
    for key in ("bbox","box","b","xywh"):
        if key in obj and isinstance(obj[key], (list,tuple)) and len(obj[key])>=4:
            x,y,w,h = obj[key][:4]
            return float(x)+float(w)/2.0, float(y)+float(h)/2.0
    if all(k in obj for k in ("x","y","w","h")):
        return float(obj["x"])+float(obj["w"])/2.0, float(obj["y"])+float(obj["h"])/2.0
    return None

def build_track_from_ann(ann_json: Path, out_csv: Path):
    data = json.loads(ann_json.read_text(encoding="utf-8"))
    frames = data.get("frames", data.get("video_frames", []))  # 兼容字段
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame","obj_id","x","y"])
        for t, fr in enumerate(frames):
            for obj in fr.get("objects", fr.get("instances", [])):
                oid = int(obj.get("id", obj.get("obj_id", 0)))
                c = get_bbox_center(obj)
                if c is None: continue
                x,y = c
                w.writerow([t, oid, x, y])

def find_or_make_track(stem: str, tracks_dir: Path, ann_dir: Path) -> Path:
    tp = match_track(tracks_dir, stem)
    if tp is not None: return tp
    # 用 JSON 生成
    if ann_dir is None or not ann_dir.exists():
        raise RuntimeError(f"No track CSV for stem={stem} and ann_dir missing.")
    digs = digits_only(stem)
    # 常见命名：video_XXXXX.json / XXXXX.json
    cands = []
    for pat in (f"{stem}.json", f"video_{stem}.json", f"*{stem}*.json"):
        cands += list(ann_dir.glob(pat))
    if digs:
        for pat in (f"{digs}.json", f"video_{digs}.json", f"*{digs}*.json"):
            cands += list(ann_dir.glob(pat))
    cands = sorted(set(cands))
    if not cands:
        raise RuntimeError(f"No track CSV and no matching annotation JSON for stem={stem}")
    out_csv = tracks_dir / f"{stem}.csv"
    build_track_from_ann(cands[0], out_csv)
    return out_csv

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
        if h < w:
            nh = short_side; nw = int(w * nh / h)
        else:
            nw = short_side; nh = int(h * nw / w)
        fr = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_AREA)
        imgs.append(fr)
    cap.release()
    return imgs, idxs

def read_track_best(csv_path: Path, min_len_obj=5):
    # 返回：frames(list), xs(np.array), ys(np.array)
    by_obj = {}
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            fr = int(float(row["frame"]))
            oid = int(float(row.get("obj_id", 0)))
            x = float(row["x"]); y = float(row["y"])
            by_obj.setdefault(oid, []).append((fr,x,y))
    if not by_obj:
        return [], np.array([]), np.array([])
    # 选帧数最多的物体
    best_oid = max(by_obj.keys(), key=lambda k: len(by_obj[k]))
    best = sorted(by_obj[best_oid], key=lambda t: t[0])
    if len(best) >= min_len_obj:
        frames = [t for (t,_,_) in best]
        xs = np.array([x for (_,x,_) in best], dtype=float)
        ys = np.array([y for (_,_,y) in best], dtype=float)
        return frames, xs, ys
    # 降级：按帧取所有物体的质心
    by_frame = {}
    for oid, seq in by_obj.items():
        for (t,x,y) in seq:
            by_frame.setdefault(t, []).append((x,y))
    frames_sorted = sorted(by_frame.keys())
    if len(frames_sorted) < 2:
        return [], np.array([]), np.array([])
    xs=[]; ys=[]
    for t in frames_sorted:
        arr = np.array(by_frame[t], dtype=float)
        xv,yv = arr.mean(axis=0)
        xs.append(xv); ys.append(yv)
    return frames_sorted, np.array(xs), np.array(ys)

def smooth_and_downsample(xs, ys, points=8, k=7):
    k = max(1, int(k)); k = k if k % 2 == 1 else k+1
    def movavg(a, k):
        if len(a) < k: return a.copy()
        pad = k//2
        ap = np.pad(a, (pad,pad), mode="edge")
        ker = np.ones(k)/k
        return np.convolve(ap, ker, mode="valid")
    xs_s = movavg(xs, k); ys_s = movavg(ys, k)
    n = min(max(6, points), 16)
    if len(xs_s) < 2:  # 极端保护
        return xs, ys
    idx = np.linspace(0, len(xs_s)-1, num=n, dtype=int)
    return xs_s[idx], ys_s[idx]

def draw_dashed_polyline(img, pts, color, thickness=2, dash=12, gap=8):
    for i in range(len(pts)-1):
        p1 = np.array(pts[i], dtype=float)
        p2 = np.array(pts[i+1], dtype=float)
        seg = p2 - p1
        L = np.linalg.norm(seg)
        if L < 1e-3: continue
        v = seg / L
        s = 0.0
        while s < L:
            e = min(s + dash, L)
            a = (p1 + v*s).astype(int)
            b = (p1 + v*e).astype(int)
            cv2.line(img, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)
            s += dash + gap

def try_one_video(vp: Path, tracks_dir: Path, ann_dir: Path, n_frames=6, short=360):
    stem = vp.stem
    tp = find_or_make_track(stem, tracks_dir, ann_dir)
    frames, xs, ys = read_track_best(tp, min_len_obj=5)
    if len(frames) < 2:
        return None, f"Track too short: {tp.name}"
    imgs, idxs = sample_frames(vp, n_frames, short)
    if not imgs:
        return None, f"No frames from {vp.name}"
    xsp, ysp = smooth_and_downsample(xs, ys, points=8, k=7)
    planned_pts = np.stack([xsp, ysp], axis=1)

    # 叠加绘制
    out_panels = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, img in zip(idxs, imgs):
        panel = img.copy()
        # planned: 虚线（蓝橙随便，这里橙色）
        draw_dashed_polyline(panel, planned_pts, (30,160,240), thickness=2, dash=12, gap=8)
        # realized: 到该帧（蓝色）
        # 找到 <= idx 的帧索引
        valid = [j for j,t in enumerate(frames) if t <= int(idx)]
        if len(valid) >= 2:
            pts = np.stack([xs[valid], ys[valid]], axis=1).astype(int)
            cv2.polylines(panel, [pts], isClosed=False, color=(255,180,80), thickness=2, lineType=cv2.LINE_AA)
        # 角标
        txt = f"{stem} | t={int(idx)}"
        (tw, th), _ = cv2.getTextSize(txt, font, 0.6, 1)
        cv2.rectangle(panel, (6,6), (6+tw+12, 6+th+12), (255,255,255), -1)
        cv2.putText(panel, txt, (12, 6+th+4), font, 0.6, (0,0,0), 1, cv2.LINE_AA)
        out_panels.append(panel)

    # 横向拼
    hmin = min(p.shape[0] for p in out_panels)
    out_panels = [cv2.resize(p, (int(p.shape[1]*hmin/p.shape[0]), hmin), interpolation=cv2.INTER_AREA) for p in out_panels]
    grid = cv2.hconcat(out_panels)
    return grid, f"OK ({vp.name})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--tracks_dir", required=True)
    ap.add_argument("--ann_dir", required=True)
    ap.add_argument("--try_top", type=int, default=20, help="最多尝试前 N 个视频，直到成功")
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--short", type=int, default=360)
    ap.add_argument("--out", type=str, default="figs/plan_vs_real_oracle.png")
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    tracks_dir = Path(args.tracks_dir)
    ann_dir = Path(args.ann_dir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 遍历前 N 个视频，直到找到可用轨迹
    vids = sorted(video_dir.glob("*.mp4"))[:max(1,args.try_top)]
    last_err = None
    for vp in vids:
        grid, msg = try_one_video(vp, tracks_dir, ann_dir, n_frames=args.n_frames, short=args.short)
        if grid is not None:
            cv2.imwrite(args.out, grid)
            print(f"[OK] saved {args.out}  using video={vp.name}")
            return
        else:
            print(f"[skip] {vp.name}: {msg}")
            last_err = msg
    raise RuntimeError(f"All {len(vids)} candidates failed. Last: {last_err}")

if __name__ == "__main__":
    main()
