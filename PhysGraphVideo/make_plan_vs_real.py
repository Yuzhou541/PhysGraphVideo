
#!/usr/bin/env python3
# Overlay planned (dashed) vs realized (solid) trajectories and contact events over sampled frames.
# Inputs:
#   --video path/to/ours/example.mp4
#   --planned plans/example.json    (format below)
#   --realized tracks/example.csv   (frame,obj_id,x,y ; pixel coords)
#   --contacts contacts/example.csv (frame,i,j)  [optional]
# Outputs:
#   figs/plan_vs_real.pdf
#
# planned JSON example:
# {
#   "width": 1280, "height": 720,
#   "objects": [{"id":1,"traj":[[x1,y1],...]], {"id":2, ...}]
# }
#
import argparse, os, csv, json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

def sample_frames(video_path, n_frames=6):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(frames-1,0), n_frames, dtype=int)
    imgs = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok: break
        imgs.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return imgs, idxs

def load_realized(csv_path):
    # returns dict: obj_id -> dict(frame -> (x,y))
    data = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fr = int(float(r["frame"])); oid = int(float(r["obj_id"]))
            x = float(r["x"]); y = float(r["y"])
            data.setdefault(oid, {})[fr] = (x,y)
    return data

def load_planned(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_contacts(csv_path):
    events = []  # list of (frame,i,j)
    if csv_path and Path(csv_path).exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                events.append((int(float(r["frame"])), int(float(r["i"])), int(float(r["j"]))))
    return events

def draw_overlay(img, t_idx, planned, realized_map, contacts_at_t):
    import matplotlib.patches as mpatches
    ax = plt.gca()
    ax.imshow(img); ax.set_axis_off()

    # planned dashed
    for obj in planned.get("objects", []):
        pts = np.array(obj.get("traj", []), dtype=float)
        if len(pts) < 2: continue
        ax.plot(pts[:,0], pts[:,1], linestyle="--", linewidth=2.0)

    # realized solid (connect past positions up to t)
    for oid, frames in realized_map.items():
        frs = sorted(k for k in frames.keys() if k <= t_idx)
        if len(frs) >= 2:
            pts = np.array([frames[k] for k in frs], dtype=float)
            ax.plot(pts[:,0], pts[:,1], linewidth=2.0)

    # contacts as markers
    for (_, i, j) in contacts_at_t:
        # highlight roughly at mid-point if both tracked
        if i in realized_map and j in realized_map and t_idx in realized_map[i] and t_idx in realized_map[j]:
            xi, yi = realized_map[i][t_idx]; xj, yj = realized_map[j][t_idx]
            mx, my = (xi+xj)/2, (yi+yj)/2
            circ = plt.Circle((mx, my), radius=max(img.shape[:2])*0.01, fill=False, linewidth=2.0)
            ax.add_patch(circ)

def main(args):
    planned = load_planned(args.planned)
    realized = load_realized(args.realized)
    contacts = load_contacts(args.contacts)
    # group contacts by frame
    contacts_by_f = {}
    for fr,i,j in contacts:
        contacts_by_f.setdefault(fr, []).append((fr,i,j))

    imgs, idxs = sample_frames(args.video, args.n_frames)
    cols = len(imgs)
    fig, axes = plt.subplots(1, cols, figsize=(min(11, 2.0*cols), 2.6))
    if cols == 1:
        axes = [axes]
    for ax, img, fr in zip(axes, imgs, idxs):
        plt.sca(ax)
        draw_overlay(img, int(fr), planned, realized, contacts_by_f.get(int(fr), []))
    plt.tight_layout(pad=0.1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] saved {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--planned", required=True)
    ap.add_argument("--realized", required=True)
    ap.add_argument("--contacts", default=None)
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--out", default="figs/plan_vs_real.pdf")
    args = ap.parse_args()
    main(args)
