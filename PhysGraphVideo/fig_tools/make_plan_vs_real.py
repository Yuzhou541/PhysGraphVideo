# -*- coding: utf-8 -*-
# fig_tools/make_plan_vs_real.py
# 在采样帧上叠加：planned(虚线) vs realized(实线) 轨迹；可选接触事件。
# 用法:
#   python fig_tools/make_plan_vs_real.py --video data/clevrer/videos/val/000001.mp4 \
#     --planned plans/000001.json --realized tracks/000001.csv --n_frames 6 --out figs/plan_vs_real_oracle.pdf
import argparse, os, csv, json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

def sample_frames(video_path, n_frames=6):
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {video_path}")
    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs=np.linspace(0,max(frames-1,0),n_frames,dtype=int)
    imgs=[]
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i))
        ok,fr=cap.read()
        if not ok: break
        imgs.append(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
    cap.release()
    return imgs, idxs

def load_realized(csv_path):
    data={}
    with open(csv_path,"r",newline="",encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            fr=int(float(row["frame"]))
            oid=int(float(row["obj_id"]))
            x=float(row["x"]); y=float(row["y"])
            data.setdefault(oid,{})[fr]=(x,y)
    return data

def load_planned(json_path):
    return json.load(open(json_path,"r",encoding="utf-8"))

def draw_overlay(ax, img, t_idx, planned, realized_map):
    ax.imshow(img); ax.set_axis_off()
    # planned 虚线
    for obj in planned.get("objects",[]):
        pts=np.array(obj.get("traj",[]),dtype=float)
        if len(pts)>=2:
            ax.plot(pts[:,0],pts[:,1],linestyle="--",linewidth=2.0)
    # realized 实线 (到当前帧为止)
    for oid, frames in realized_map.items():
        frs=sorted(k for k in frames.keys() if k<=t_idx)
        if len(frs)>=2:
            pts=np.array([frames[k] for k in frs],dtype=float)
            ax.plot(pts[:,0],pts[:,1],linewidth=2.0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--planned", required=True)
    ap.add_argument("--realized", required=True)
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--out", default="figs/plan_vs_real.pdf")
    args=ap.parse_args()

    planned=load_planned(args.planned)
    realized=load_realized(args.realized)
    imgs, idxs = sample_frames(args.video, args.n_frames)

    cols=len(imgs)
    fig,axes=plt.subplots(1,cols,figsize=(min(11,2.0*cols),2.6))
    if cols==1: axes=[axes]
    for ax,img,fr in zip(axes, imgs, idxs):
        draw_overlay(ax, img, int(fr), planned, realized)
    plt.tight_layout(pad=0.1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out,bbox_inches="tight",pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] saved {args.out}")

if __name__=="__main__":
    main()
