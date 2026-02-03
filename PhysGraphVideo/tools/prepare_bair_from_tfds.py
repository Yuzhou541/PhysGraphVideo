# tools/prepare_bair_from_tfds.py
import os, argparse, math, shutil
from pathlib import Path

def _lazy_import_tfds():
    try:
        import tensorflow_datasets as tfds
        import tensorflow as tf
        return tfds, tf
    except Exception as e:
        raise RuntimeError(
            "TensorFlow Datasets 未安装。请先在当前conda环境安装：\n"
            "  pip install tensorflow-cpu==2.17.0 tensorflow-datasets==4.9.4\n"
            "（仅用于下载与导出，安装后可继续使用PyTorch训练）"
        ) from e

def save_npz(seq, out_dir, base, begin, end):
    import numpy as np
    T = end - begin
    vid_thwc = seq[begin:end]                 # [T, H, W, C], uint8
    # 统一为 [T, C, H, W], uint8（后续Dataset会转float/归一化）
    vid_tchw = vid_thwc.transpose(0, 3, 1, 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{base}_{begin:06d}_{end:06d}.npz", video=vid_tchw)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="data/bair",
                   help="导出根目录，最终结构为 root/{train,val,test}/*.npz")
    p.add_argument("--splits", type=str, default="train,val,test",
                   help="要导出的TFDS划分，逗号分隔")
    p.add_argument("--t_in", type=int, default=2, help="输入帧数")
    p.add_argument("--t_out", type=int, default=14, help="预测帧数")
    p.add_argument("--stride", type=int, default=1, help="滑窗步长")
    p.add_argument("--image_size", type=int, default=64, help="输出分辨率（64 与论文对齐）")
    p.add_argument("--format", type=str, default="npz", choices=["npz"],
                   help="导出格式，目前固定为npz")
    args = p.parse_args()

    tfds, tf = _lazy_import_tfds()

    root = Path(args.root)
    T_tot = args.t_in + args.t_out
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    # TFDS 数据集名（小规模64x64版本，学术界广泛使用）
    ds_name = "bair_robot_pushing_small"

    for split in splits:
        print(f"[BAIR] 读取 TFDS: {ds_name}/{split}")
        ds = tfds.load(ds_name, split=split, as_supervised=False)
        np_ds = tfds.as_numpy(ds)

        out_dir = root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        num_vid, num_seq = 0, 0
        for ex in np_ds:
            # TFDS样本字段：'video'，形状约 [30, 64, 64, 3]，uint8
            video = ex["video"]  # THWC, uint8
            H, W, C = video.shape[1], video.shape[2], video.shape[3]
            assert C in (1, 3), f"unexpected channels={C}"

            # 若尺寸不是目标分辨率，可选择跳过或插值。TFDS small默认64，这里直接断言：
            assert (H, W) == (args.image_size, args.image_size), \
                f"found {(H,W)}, expected {(args.image_size,args.image_size)}"

            # 滑动窗口导出
            T = video.shape[0]
            if T < T_tot:
                continue
            base = f"vid{num_vid:06d}"
            for begin in range(0, T - T_tot + 1, args.stride):
                end = begin + T_tot
                save_npz(video, out_dir, base, begin, end)
                num_seq += 1
            num_vid += 1

        print(f"[BAIR] 导出完成：split={split} videos={num_vid} seq(npz)={num_seq} -> {out_dir}")

if __name__ == "__main__":
    main()
