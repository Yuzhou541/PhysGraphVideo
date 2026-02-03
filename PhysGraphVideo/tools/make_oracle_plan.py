# -*- coding: utf-8 -*-
# tools/make_oracle_plan.py
# 用法:
#   python tools/make_oracle_plan.py tracks/000001.csv plans/000001.json --W 1280 --H 720 --points 8 --smooth 7
import os, sys, csv, json, argparse
import numpy as np

def moving_average(x, k):
    k=max(1,int(k)); k = k if k%2==1 else k+1  # odd
    pad=k//2
    if len(x)<k: return x.copy()
    xpad=np.pad(x,(pad,pad),mode="edge")
    kern=np.ones(k)/k
    return np.convolve(xpad,kern,mode="valid")

def main(args):
    by_id={}
    with open(args.csv,"r",encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            fr=int(float(row["frame"]))
            oid=int(float(row["obj_id"]))
            x=float(row["x"]); y=float(row["y"])
            by_id.setdefault(oid,[]).append((fr,x,y))
    objects=[]
    for oid,pts in by_id.items():
        pts.sort(key=lambda t:t[0])
        xs=np.array([p[1] for p in pts],dtype=float)
        ys=np.array([p[2] for p in pts],dtype=float)
        xs=moving_average(xs,args.smooth)
        ys=moving_average(ys,args.smooth)
        n=min(max(6,args.points),16)
        idx=np.linspace(0,len(xs)-1,num=n,dtype=int)
        traj=[[float(xs[i]),float(ys[i])] for i in idx]
        objects.append({"id":int(oid),"traj":traj})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"width":args.W,"height":args.H,"objects":objects}, open(args.out,"w",encoding="utf-8"))
    print(f"[OK] saved {args.out}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out")
    ap.add_argument("--W", type=int, default=1280)
    ap.add_argument("--H", type=int, default=720)
    ap.add_argument("--points", type=int, default=8)
    ap.add_argument("--smooth", type=int, default=7)
    args=ap.parse_args()
    main(args)
