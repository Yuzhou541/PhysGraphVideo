# -*- coding: utf-8 -*-
# tools/clevrer_json_to_tracks.py
# 用法:
#   python tools/clevrer_json_to_tracks.py val
# 从 data/clevrer/annotations/<split>/*.json 提取每帧每物体 bbox 中心 -> tracks/<stem>.csv
import os, sys, json, csv, glob

split = sys.argv[1] if len(sys.argv) > 1 else "val"
ann_dir = os.path.join("data","clevrer","annotations",split)
out_dir = "tracks"
os.makedirs(out_dir, exist_ok=True)

def get_bbox_center(obj):
    for key in ("bbox","box","b","xywh"):
        if key in obj and isinstance(obj[key], (list,tuple)) and len(obj[key])>=4:
            x,y,w,h = obj[key][:4]
            return float(x)+float(w)/2.0, float(y)+float(h)/2.0
    if all(k in obj for k in ("x","y","w","h")):
        return float(obj["x"])+float(obj["w"])/2.0, float(obj["y"])+float(obj["h"])/2.0
    return None

count=0
for jp in glob.glob(os.path.join(ann_dir,"*.json")):
    stem=os.path.splitext(os.path.basename(jp))[0]
    data=json.load(open(jp,"r",encoding="utf-8"))
    frames=data.get("frames",[])
    out_csv=os.path.join(out_dir,f"{stem}.csv")
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["frame","obj_id","x","y"])
        for t,fr in enumerate(frames):
            for obj in fr.get("objects",[]):
                oid=int(obj.get("id", obj.get("obj_id", 0)))
                c=get_bbox_center(obj)
                if c is None: continue
                x,y=c
                w.writerow([t,oid,x,y])
    count+=1
print(f"[OK] wrote {count} CSV files -> {out_dir}")
