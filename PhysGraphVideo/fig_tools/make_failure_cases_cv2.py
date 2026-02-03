# -*- coding: utf-8 -*-
# fig_tools/make_failure_cases_cv2.py
# 无重叠：更高卡片、自动折行、右上角徽章避让标题区
# 输出: figs/failure_cases.png

import argparse, os, textwrap
import numpy as np
import cv2
from pathlib import Path

def put_multiline(img, text, x, y, width_px, scale=0.55, color=(30,30,30), bold=False, line_gap=4):
    approx_char_w = max(8, int(10 * scale))  # 字宽估计
    max_chars = max(8, width_px // approx_char_w)
    lines = textwrap.wrap(text, width=max_chars)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i*int(16*scale+line_gap)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if bold else 1, cv2.LINE_AA)
    return y + len(lines)*int(16*scale+line_gap)

def draw_badge(img, text, topright, color_bg=(240,248,255), color_fg=(30,60,90), scale=0.48):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x2, y1 = topright
    x1 = x2 - tw - 14
    y2 = y1 + th + 12
    cv2.rectangle(img, (x1,y1), (x2,y2), color_bg, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1,y1), (x2,y2), (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(img, text, (x1+7, y1+th+1-2), cv2.FONT_HERSHEY_SIMPLEX, scale, color_fg, 1, cv2.LINE_AA)

CASES = [
    ("Ambiguous Contact",
     "Unclear touching order when objects overlap; contact mis-timing.",
     "Mitigation: contact-order loss; larger K; mask gap."),
    ("Multi-body Chain",
     "Long reaction chains drift; late collisions under-modeled.",
     "Mitigation: late schedule; flow temporal weight; replay."),
    ("Thin/SDF Failure",
     "Thin parts vanish in SDF; false penetration alarms.",
     "Mitigation: SDF fine-tune; thickness prior; hi-res mask."),
    ("Fast Camera Pan",
     "Large apparent accelerations bias gravity/tilt estimation.",
     "Mitigation: IMU/vanishing-line tilt prior; robust MAE."),
    ("Long-horizon Drift",
     "Appearance gradually changes beyond 20s; ID inconsistency.",
     "Mitigation: distillation+EWC; periodic key-frames."),
    ("Fluid-like Motion",
     "Non-rigid flows break simple smoothness; shimmering.",
     "Mitigation: local flow curvature loss; adaptive smoothness.")
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/failure_cases.png")
    ap.add_argument("--W", type=int, default=1700)
    ap.add_argument("--H", type=int, default=980)  # 加高避免重叠
    args = ap.parse_args()

    Path("figs").mkdir(parents=True, exist_ok=True)
    img = np.full((args.H, args.W, 3), 255, np.uint8)

    # 标题
    cv2.putText(img, "Failure cases and mitigations", (26, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20,20,20), 2, cv2.LINE_AA)

    rows, cols = 2, 3
    pad = 26
    top = 74
    # 卡片高度更大，内部留白更大
    cell_w = (args.W - pad*(cols+1)) // cols
    cell_h = (args.H - top - pad*(rows+1)) // rows

    for idx, (title, symptom, fix) in enumerate(CASES):
        r = idx // cols
        c = idx % cols
        x = pad + c*(cell_w + pad)
        y = top + r*(cell_h + pad)

        # 卡片背景
        cv2.rectangle(img, (x, y), (x+cell_w, y+cell_h), (238,238,238), -1, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x+cell_w, y+cell_h), (210,210,210), 1, cv2.LINE_AA)

        # 缩略图区域（占上半区）
        thumb_h = int(cell_h*0.50)
        cv2.rectangle(img, (x+12, y+12), (x+cell_w-12, y+12+thumb_h), (250,250,250), -1, cv2.LINE_AA)
        cv2.rectangle(img, (x+12, y+12), (x+cell_w-12, y+12+thumb_h), (205,205,205), 1, cv2.LINE_AA)
        # 填充几条抽象线防空白
        for k in range(7):
            x1 = x+24 + k*int((cell_w-48)/7)
            y1 = y+18 + (k%3)*14
            x2 = x1 + int((cell_w-80)/7)
            y2 = y1 + 10 + (k%2)*8
            cv2.line(img, (x1,y1), (x2,y2), (210,210,210), 2, cv2.LINE_AA)

        # 右上角徽章（严格放在缩略图区域右上，避开标题）
        draw_badge(img, fix, (x+cell_w-14, y+16))

        # 标题（加粗，自动折行）
        y_cursor = y + 12 + thumb_h + 28
        y_cursor = put_multiline(img, title, x+16, y_cursor, cell_w-32, scale=0.72, bold=True)
        y_cursor += 6
        # 症状（折行）
        put_multiline(img, symptom, x+16, y_cursor, cell_w-32, scale=0.58, color=(50,50,50))

    cv2.imwrite(args.out, img)
    print(f"[OK] saved {args.out}")

if __name__ == "__main__":
    main()
