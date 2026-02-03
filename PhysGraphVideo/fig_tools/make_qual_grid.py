# -*- coding: utf-8 -*-
# fig_tools/make_qual_grid.py
# 质性网格：行=方法，列=等距采样帧
# 用法(示例):
#   python fig_tools/make_qual_grid.py --methods ref:data/clevrer/videos/val \
#       --stems lists/qual_list.txt --n_frames 6 --resize 360 --out figs/qual_grid_ref.pdf
import argparse, os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

def list_videos(d):
    vids=[]
    for ext in ("*.mp4","*.webm","*.avi","*.mov","*.mkv","*.gif"):
        vids+=sorted(Path(d).glob(ext))
    return vids

def sample_frames(video, n, resize_short=None):
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise RuntimeError(f"open fail: {video}")
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs=np.linspace(0,max(total-1,0),n,dtype=int)
    imgs=[]
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i))
        ok,fr=cap.read()
        if not ok: break
        fr=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
        if resize_short:
            h,w=fr.shape[:2]
            if h<w: nh=resize_short; nw=int(w*nh/h)
            else:  nw=resize_short; nh=int(h*nw/w)
            fr=cv2.resize(fr,(nw,nh),interpolation=cv2.INTER_AREA)
        imgs.append(fr)
    cap.release()
    return imgs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True, help="name:dir ...  例: ref:data/clevrer/videos/val ours:runs/ours")
    ap.add_argument("--stems", type=str, default=None, help="可选: 文本文件, 每行一个不带扩展名的文件名")
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--resize", type=int, default=360, help="短边统一到该像素")
    ap.add_argument("--out", type=str, default="figs/qual_grid.pdf")
    ap.add_argument("--pick", type=int, default=6)
    args=ap.parse_args()

    md={}
    for item in args.methods:
        name, d = item.split(":",1)
        md[name]=d

    # 选择样例
    if args.stems:
        stems=[ln.strip() for ln in open(args.stems,encoding="utf-8") if ln.strip()]
    else:
        stems=[p.stem for p in list_videos(list(md.values())[0])]
    stems=stems[:args.pick]

    methods=list(md.keys())
    rows=len(methods)*len(stems)
    cols=args.n_frames

    fig,axes=plt.subplots(rows, cols, figsize=(10, max(3,1.2*rows+0.6)))
    if rows==1: 
        import numpy as np
        axes=np.expand_dims(axes,0)
    if cols==1:
        import numpy as np
        axes=np.expand_dims(axes,1)

    r=0
    for stem in stems:
        for m in methods:
            # 找到对应视频
            vp=None
            for ext in ("mp4","webm","avi","mov","mkv","gif"):
                cand=Path(md[m])/f"{stem}.{ext}"
                if cand.exists(): vp=cand; break
            if vp is None: raise RuntimeError(f"missing video: {m}:{stem}")
            imgs=sample_frames(vp, args.n_frames, args.resize)
            for c,img in enumerate(imgs):
                ax=axes[r,c]; ax.imshow(img); ax.set_axis_off()
                if c==0:
                    ax.text(0.01,0.98,m,fontsize=9,fontweight="bold",ha="left",va="top",
                            transform=ax.transAxes,bbox=dict(facecolor="white",alpha=0.7,edgecolor="none",pad=2.5))
        for c in range(cols):
            axes[r,c].text(0.5,1.02,stem,fontsize=9,ha="center",va="bottom",transform=axes[r,c].transAxes)
        r+=len(methods)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout(pad=0.1)
    fig.savefig(args.out,bbox_inches="tight",pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] saved {args.out}")

if __name__=="__main__":
    main()
