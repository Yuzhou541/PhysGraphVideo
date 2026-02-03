# -*- coding: utf-8 -*-
# fig_tools/make_discussion_sensitivity_cv2.py
# 纯 OpenCV 画两栏灵敏度图，加入防重叠布局（更大边距、两行x轴标签、图例区域固定）
# 输出: figs/discussion_sensitivity.png

import argparse, os
import numpy as np
import cv2
from pathlib import Path

def text_size(text, scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    return tw, th

def put_text(img, text, org, scale=0.5, color=(30,30,30), thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_axes(img, rect, x_labels, y_min, y_max, title=None, x_label=None, y_label=None):
    """画坐标轴并返回绘图区域，rect=(x0,y0,w,h)。带更多上下留白避免重叠。"""
    x0,y0,w,h = rect
    pad_l, pad_r, pad_t, pad_b = 72, 28, 56, 72  # 加大留白
    # 背景
    cv2.rectangle(img, (x0,y0), (x0+w, y0+h), (246,246,246), -1)
    # 绘图区
    px0, py0 = x0+pad_l, y0+pad_t
    pw, ph = w-pad_l-pad_r, h-pad_t-pad_b

    # 网格
    for t in np.linspace(0,1,5):
        y = int(py0 + ph - t*ph)
        cv2.line(img, (px0,y), (px0+pw,y), (230,230,230), 1, cv2.LINE_AA)
    # 外框
    cv2.rectangle(img, (px0,py0), (px0+pw, py0+ph), (150,150,150), 1, cv2.LINE_AA)

    # X 刻度（两行布局避免重叠）
    n = len(x_labels)
    for i, lab in enumerate(x_labels):
        x = int(px0 + (0 if n==1 else i*pw/(n-1)))
        cv2.line(img, (x, py0+ph), (x, py0+ph+5), (0,0,0), 1, cv2.LINE_AA)
        # 两行：过长标签切分或上下两行交错
        t1, t2 = lab, ""
        if len(lab) > 10 and " " in lab:
            mid = lab.find(" ")
            t1, t2 = lab[:mid], lab[mid+1:]
        y_base = py0+ph+22
        put_text(img, t1, (x - text_size(t1,0.5)[0]//2, y_base), 0.5)
        if t2:
            put_text(img, t2, (x - text_size(t2,0.5)[0]//2, y_base+18), 0.5)
        else:
            # 交错：偶数行向下偏移，减少相邻碰撞概率
            if (i % 2)==1:
                put_text(img, "", (x, y_base+18), 0.5)

    # Y 刻度
    for v in np.linspace(y_min, y_max, 5):
        t = (v - y_min) / max(1e-6, (y_max - y_min))
        y = int(py0 + ph - t*ph)
        lab = f"{v:.0f}"
        put_text(img, lab, (px0-48, y+5), 0.5)

    # 标题与轴标签
    if title:
        put_text(img, title, (x0+8, y0+26), 0.68, (20,20,20), 2)
    if x_label:
        put_text(img, x_label, (px0+pw-90, py0+ph+48), 0.5)
    if y_label:
        put_text(img, y_label, (px0-64, py0-10), 0.5)
    return (px0, py0, pw, ph)

def plot_line(img, plot_rect, ys, color, label=None, y_min=0, y_max=1, marker=True, legend_anchor=None):
    px0, py0, pw, ph = plot_rect
    n = len(ys)
    pts = []
    for i,y in enumerate(ys):
        x = int(px0 + (0 if n==1 else i*pw/(n-1)))
        t = (y - y_min) / max(1e-6, (y_max - y_min))
        yy = int(py0 + ph - t*ph)
        pts.append((x, yy))
    for i in range(len(pts)-1):
        cv2.line(img, pts[i], pts[i+1], color, 2, cv2.LINE_AA)
    if marker:
        for (x,y) in pts:
            cv2.circle(img, (x,y), 4, color, -1, cv2.LINE_AA)

    if label and legend_anchor is not None:
        lx, ly, lw, lh = legend_anchor
        # 固定左上角图例框，纵向不重叠
        yrow = plot_line.legend_row * 22 + 10
        cv2.rectangle(img, (lx+8, ly+8+yrow-12), (lx+8+18, ly+8+yrow-4), color, -1)
        put_text(img, label, (lx+8+26, ly+8+yrow-4), 0.5)
        plot_line.legend_row += 1
plot_line.legend_row = 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/discussion_sensitivity.png")
    ap.add_argument("--W", type=int, default=1700)
    ap.add_argument("--H", type=int, default=760)
    args = ap.parse_args()

    Path("figs").mkdir(parents=True, exist_ok=True)
    canvas = np.full((args.H, args.W, 3), 255, np.uint8)

    # 数据（与你论文表格一致）
    Ks = [0,5,10,20,40]
    physics = [72.8, 80.2, 84.7, 83.9, 82.4]
    planfaith = [71.0, 82.0, 85.0, 84.0, 81.0]

    sched_labels = ["early", "mid", "late (ours)"]
    physics_s = [81.6, 82.3, 84.7]
    planfaith_s = [80.0, 82.0, 85.0]

    # 两栏区域（加宽留白）
    left_rect  = (40,  70, args.W//2 - 60, args.H - 110)
    right_rect = (args.W//2 + 20, 70, args.W//2 - 60, args.H - 110)

    # Panel (a)
    plot_line.legend_row = 0
    pr = draw_axes(canvas, left_rect, [str(k) for k in Ks], 70, 88,
                   title="(a) Retrieval top-K",
                   x_label="K", y_label="score (%)")
    # 固定图例锚点，防止线条与文字碰撞
    legend_anchor = (pr[0], pr[1], 160, 80)
    plot_line(canvas, pr, physics, (40,140,40), "Physics adherence", 70, 88, legend_anchor=legend_anchor)
    plot_line(canvas, pr, planfaith, (40,90,205), "Plan faithfulness", 70, 88, legend_anchor=legend_anchor)

    # Panel (b)
    plot_line.legend_row = 0
    pr2 = draw_axes(canvas, right_rect, sched_labels, 78, 88,
                    title="(b) Physics loss schedule",
                    x_label="schedule", y_label="score (%)")
    legend_anchor2 = (pr2[0], pr2[1], 160, 80)
    plot_line(canvas, pr2, physics_s, (40,140,40), "Physics adherence", 78, 88, legend_anchor=legend_anchor2)
    plot_line(canvas, pr2, planfaith_s, (40,90,205), "Plan faithfulness", 78, 88, legend_anchor=legend_anchor2)

    # 总标题（顶部左侧，避免压到图）
    put_text(canvas, "Sensitivity: Retrieval Top-K and Physics Schedule", (40, 44), 0.85, (20,20,20), 2)

    cv2.imwrite(args.out, canvas)
    print(f"[OK] saved {args.out}")

if __name__ == "__main__":
    main()
