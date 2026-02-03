# 文件路径：tools/clevrer_flatten.py
# 用法：python -m tools.clevrer_flatten  [--root data/clevrer]
import argparse, os, shutil, glob
from pathlib import Path

def _flatten_split(root_dir: Path, sub: str, split: str, exts):
    src = root_dir / sub / split
    if not src.exists():
        return 0
    flat_tmp = src.with_name(src.name + "_flat")
    flat_tmp.mkdir(parents=True, exist_ok=True)
    n = 0
    for ext in exts:
        for p in src.rglob(f"*{ext}"):
            if p.parent == src:
                continue
            dst = flat_tmp / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
                n += 1
    # 把已在顶层的也搬到 flat_tmp，保证全集
    for ext in exts:
        for p in src.glob(f"*{ext}"):
            dst = flat_tmp / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
                n += 1
    # 备份原目录并替换为扁平目录
    bak = src.with_name(src.name + "_bak")
    if bak.exists():
        shutil.rmtree(bak)
    src.rename(bak)
    flat_tmp.rename(src)
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/clevrer", type=str)
    args = ap.parse_args()
    root = Path(args.root)

    if not root.exists():
        raise SystemExit(f"[ERR] root not found: {root.resolve()}")

    print(f"[CLEVRER] flatten videos/annotations under: {root.resolve()}")

    total_v = 0
    for split in ["train", "val", "test"]:
        total_v += _flatten_split(root, "videos", split, (".mp4", ".avi", ".mkv"))

    total_a = 0
    for split in ["train", "val"]:
        total_a += _flatten_split(root, "annotations", split, (".json", ".txt"))

    def _count(pat): return len(glob.glob(str(root / pat)))
    print(f"[DONE] videos: train={_count('videos/train/*.mp4')}, "
          f"val={_count('videos/val/*.mp4')}, test={_count('videos/test/*.mp4')}")
    print(f"[DONE] annos : train={_count('annotations/train/*.json')}, "
          f"val={_count('annotations/val/*.json')}")
    print("[TIP] 原分段目录已备份为 *_bak，若需还原可手动恢复。")

if __name__ == "__main__":
    main()
