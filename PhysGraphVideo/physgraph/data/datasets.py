# -*- coding: utf-8 -*-
"""
Unified dataloaders for multiple video datasets (frames dir / video files / npz).
Each __getitem__ returns (inp, tgt):
  inp: [T_in,  C, H, W]  float32 in [0,1]
  tgt: [T_out, C, H, W]  float32 in [0,1]
"""

import os, json, random, glob
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2


# ---------------------------
# Utilities
# ---------------------------

def _as_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return x

def _imread_any(fp, hw:Tuple[int,int], channels:int) -> np.ndarray:
    H, W = hw
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {fp}")
    if img.ndim == 2:
        img = img[..., None]
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    if channels == 1 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None]
    elif channels == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    img = img.astype(np.float32) / 255.0
    return img  # H,W,C

def _frames_from_video(mp4_path:str, hw:Tuple[int,int], channels:int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {mp4_path}")
    frames = []
    ok = True
    while ok:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 2:
            frame = frame[..., None]
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        frame = cv2.resize(frame, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
        if channels == 1 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[..., None]
        elif channels == 3 and frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        frames.append(frame.astype(np.float32)/255.0)
    cap.release()
    return frames

def _hwc_to_tchw(frames: List[np.ndarray]) -> torch.Tensor:
    if len(frames) == 0:
        return torch.empty(0)
    arr = np.stack(frames, axis=0)  # T,H,W,C
    arr = np.transpose(arr, (0,3,1,2))  # T,C,H,W
    return torch.from_numpy(arr.astype(np.float32))

def _ensure_channels(frames: np.ndarray, channels:int) -> np.ndarray:
    if frames.ndim != 4:
        raise ValueError("frames must be (T,H,W,C)")
    if channels == frames.shape[-1]:
        return frames
    if channels == 1 and frames.shape[-1] == 3:
        gray = np.dot(frames[...,:3], [0.2989, 0.5870, 0.1140])[..., None]
        return gray.astype(np.float32)
    if channels == 3 and frames.shape[-1] == 1:
        return np.repeat(frames, 3, axis=-1)
    if channels < frames.shape[-1]:
        return frames[...,:channels]
    reps = channels // frames.shape[-1] + (1 if channels % frames.shape[-1] else 0)
    tiled = np.concatenate([frames]*reps, axis=-1)
    return tiled[...,:channels]

def _list_seq_dirs(root:str, split:str, exts=('*.png','*.jpg','*.jpeg')) -> List[str]:
    # ---- robust to wrong types in config ----
    root = _as_str(root); split = _as_str(split)
    seq_root = os.path.join(root, split)
    if not os.path.isdir(seq_root): return []
    cand = []
    for d in sorted(glob.glob(os.path.join(seq_root, '*'))):
        if os.path.isdir(d):
            for e in exts:
                if glob.glob(os.path.join(d, e)):
                    cand.append(d); break
    return cand

def _list_videos(root:str, split:str, exts=('.mp4', '.avi', '.mov', '.mkv')) -> List[str]:
    # ---- robust to wrong types in config ----
    root = _as_str(root); split = _as_str(split)
    vid_root = os.path.join(root, split)
    if not os.path.isdir(vid_root): return []
    res = []
    for e in exts:
        res += sorted(glob.glob(os.path.join(vid_root, f'*{e}')))
    return res


# ---------------------------
# Base window dataset
# ---------------------------

class BaseWindowDataset(Dataset):
    """
    Build temporal windows from sequence-dirs of frames or video files or npz (BAIR).
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        T_in: int = 16,
        T_out: int = 16,
        size: int = 64,
        channels: int = 3,
        stride: int = 1,
        min_frames: Optional[int] = None,
        accept_npz: bool = False,
        npz_filename: Optional[str] = None,
        seed: int = 0
    ):
        super().__init__()
        # ---- robust casts ----
        self.root = _as_str(root)
        self.split = _as_str(split)

        self.T_in = int(T_in)
        self.T_out = int(T_out)
        self.T_tot = self.T_in + self.T_out
        self.H = int(size); self.W = int(size); self.C = int(channels)
        self.stride = max(1, int(stride))
        self.min_frames = self.T_tot if min_frames is None else int(min_frames)
        self.accept_npz = bool(accept_npz)
        self.npz_filename = _as_str(npz_filename) if npz_filename is not None else None
        self.rng = random.Random(int(seed))

        self.seq_dirs = _list_seq_dirs(self.root, self.split)
        self.vid_files = _list_videos(self.root, self.split)
        self.samples = []  # (kind, ref, start)

        self.npz_data = None
        if self.accept_npz:
            fn = self.npz_filename or f"{self.split}.npz"
            npz_path = os.path.join(self.root, fn)
            if os.path.isfile(npz_path):
                self.npz_data = np.load(npz_path)
                key = 'videos' if 'videos' in self.npz_data.files else None
                if key is None:
                    for cand in ['xs', 'frames', 'data', 'arr_0']:
                        if cand in self.npz_data.files:
                            key = cand; break
                if key is None:
                    raise ValueError(f"NPZ at {npz_path} has no recognized video array key.")
                vids = self.npz_data[key]
                n, T = vids.shape[0], vids.shape[1]
                for i in range(n):
                    if T >= self.min_frames:
                        for st in range(0, T - self.T_tot + 1, self.stride):
                            self.samples.append(('npz', i, st))

        for d in self.seq_dirs:
            files = []
            for e in ('*.png','*.jpg','*.jpeg'):
                files += sorted(glob.glob(os.path.join(d, e)))
            if len(files) >= self.min_frames:
                for st in range(0, len(files)-self.T_tot+1, self.stride):
                    self.samples.append(('dir', d, st))

        for vf in self.vid_files:
            cap = cv2.VideoCapture(vf)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n >= self.min_frames:
                for st in range(0, n-self.T_tot+1, self.stride):
                    self.samples.append(('vid', vf, st))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"[{self.__class__.__name__}] No samples found. "
                f"root={self.root} split={self.split} T_tot={self.T_tot} "
                f"npz_accept={self.accept_npz} dirs={len(self.seq_dirs)} videos={len(self.vid_files)}"
            )

    def __len__(self):
        return len(self.samples)

    def _load_npz_window(self, idx:int, start:int) -> torch.Tensor:
        key = 'videos' if 'videos' in self.npz_data.files else None
        if key is None:
            for cand in ['xs','frames','data','arr_0']:
                if cand in self.npz_data.files:
                    key = cand; break
        vids = self.npz_data[key]  # [N, T, H, W, C]
        v = vids[idx, start:start+self.T_tot]
        if v.ndim == 3: v = v[..., None]
        if v.dtype not in (np.float32, np.float64):
            v = v.astype(np.float32) / 255.0
        out = []
        for t in range(v.shape[0]):
            frm = v[t]
            frm = cv2.resize(frm, (self.W, self.H), interpolation=cv2.INTER_AREA)
            frm = _ensure_channels(frm[None,...], self.C)[0]
            out.append(frm)
        return _hwc_to_tchw(out)

    def _load_dir_window(self, d:str, start:int) -> torch.Tensor:
        frames, files = [], []
        for e in ('*.png','*.jpg','*.jpeg'):
            files += sorted(glob.glob(os.path.join(d, e)))
        sl = files[start:start+self.T_tot]
        for fp in sl:
            frames.append(_imread_any(fp, (self.H,self.W), self.C))
        return _hwc_to_tchw(frames)

    def _load_vid_window(self, vf:str, start:int) -> torch.Tensor:
        frames = _frames_from_video(vf, (self.H,self.W), self.C)
        sl = frames[start:start+self.T_tot]
        return _hwc_to_tchw(sl)

    def __getitem__(self, index):
        kind, ref, st = self.samples[index]
        if kind == 'npz':
            seq = self._load_npz_window(ref, st)
        elif kind == 'dir':
            seq = self._load_dir_window(ref, st)
        elif kind == 'vid':
            seq = self._load_vid_window(ref, st)
        else:
            raise ValueError(f"Unknown sample kind: {kind}")

        inp = seq[:self.T_in]                      # [T_in,  C,H,W]
        tgt = seq[self.T_in:self.T_in+self.T_out]  # [T_out, C,H,W]
        return inp, tgt


# ---------------------------
# Specific datasets
# ---------------------------

class BouncingBalls(BaseWindowDataset):
    """Assumes pre-rendered frames under root/<split>/<seq_id>/*.png"""
    pass

class BAIRPushDataset(BaseWindowDataset):
    """Supports frames dir or NPZ with key 'videos' (or xs/frames/data/arr_0)."""
    def __init__(self, root:str, split='train', T_in=16, T_out=16, size=64, channels=3, stride=1, seed=0):
        super().__init__(root, split, T_in, T_out, size, channels, stride,
                         min_frames=None, accept_npz=True, npz_filename=None, seed=seed)

class CLEVRERDataset(BaseWindowDataset):
    """CLEVRER videos (+optional annotations under root/annotations/<split>.json)."""
    def __init__(self, root:str, split='train', T_in=8, T_out=8, size=128, channels=3, stride=1, seed=0):
        super().__init__(root, split, T_in, T_out, size, channels, stride,
                         min_frames=None, accept_npz=False, seed=seed)
        ann_path = os.path.join(self.root, 'annotations', f'{self.split}.json')
        self.annotations: Dict[str, dict] = {}
        if os.path.isfile(ann_path):
            try:
                with open(ann_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        base = os.path.splitext(os.path.basename(k))[0]
                        self.annotations[base] = v
            except Exception:
                self.annotations = {}

class MovieBenchDataset(BaseWindowDataset):
    """Long video windows for evaluation."""
    def __init__(self, root:str, split='val', T_in=16, T_out=16, size=256, channels=3, stride=4, seed=0):
        super().__init__(root, split, T_in, T_out, size, channels, stride,
                         min_frames=None, accept_npz=False, seed=seed)

class VideoBenchDataset(BaseWindowDataset):
    """Eval-oriented, human-aligned benchmark."""
    def __init__(self, root:str, split='val', T_in=16, T_out=16, size=256, channels=3, stride=4, seed=0):
        super().__init__(root, split, T_in, T_out, size, channels, stride,
                         min_frames=None, accept_npz=False, seed=seed)


# ---------------------------
# Public factory
# ---------------------------

_NAME2CLS = {
    'bouncingballs': BouncingBalls,
    'bair': BAIRPushDataset,
    'clevrer': CLEVRERDataset,
    'moviebench': MovieBenchDataset,
    'videobench': VideoBenchDataset,
}

def make_dataset(
    name: str,
    root: str,
    split: str,
    T_in: int,
    T_out: int,
    size: int,
    channels: int,
    stride: int,
    seed: int = 0,
):
    key = name.lower()
    if key not in _NAME2CLS:
        raise ValueError(f"Unknown dataset name '{name}'. Valid: {list(_NAME2CLS.keys())}")
    DS = _NAME2CLS[key]
    # ---- robust casts for join() safety ----
    root = _as_str(root); split = _as_str(split)
    return DS(root=root, split=split, T_in=T_in, T_out=T_out, size=size,
              channels=channels, stride=stride, seed=seed)

def make_dataloader(
    name: str,
    root: str,
    split: str,
    T_in: int = 16,
    T_out: int = 16,
    size: int = 64,
    channels: int = 3,
    stride: int = 1,
    batch: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: Optional[bool] = None,
    seed: int = 0,
    # ---- compatibility aliases ----
    pred_len: Optional[int] = None,   # alias of T_out
    obs_len: Optional[int]  = None,   # alias of T_in
    seq_len: Optional[int]  = None,   # accepted (unused internally)
    **extra,                           # swallow any other legacy keywords
):
    """
    Build DataLoader. Accepts legacy aliases:
      - obs_len -> T_in
      - pred_len -> T_out
      - seq_len (unused here; kept for upstream code that expects it)
    Any additional unexpected kwargs are ignored via **extra.
    """
    if obs_len is not None:
        T_in = int(obs_len)
    if pred_len is not None:
        T_out = int(pred_len)
    if shuffle is None:
        shuffle = (split == 'train')

    # (Optional) sanity: if seq_len is given and smaller than T_in+T_out, clip T_out
    if seq_len is not None:
        try:
            seq_len = int(seq_len)
            if T_in + T_out > seq_len:
                T_out = max(1, seq_len - T_in)
        except Exception:
            pass

    ds = make_dataset(
        name=name, root=root, split=split, T_in=T_in, T_out=T_out,
        size=size, channels=channels, stride=stride, seed=seed
    )
    dl = DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    return dl
