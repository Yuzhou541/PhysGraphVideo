import os, torch
from torch.utils.tensorboard import SummaryWriter
class Saver:
    def __init__(self, ckpt_dir, keep=3):
        self.dir = ckpt_dir; os.makedirs(self.dir, exist_ok=True)
        self.keep = keep; self.saved = []
    def save(self, model, step, is_best=False):
        p = os.path.join(self.dir, f"ckpt_{step}.pt")
        torch.save(model.state_dict(), p)
        self.saved.append(p)
        while len(self.saved) > self.keep:
            old = self.saved.pop(0)
            try: os.remove(old)
            except: pass
        if is_best:
            torch.save(model.state_dict(), os.path.join(self.dir, "best.pt"))
def get_writer(dir): return SummaryWriter(dir)
def write_log(writer, scalars, step, prefix=""):
    for k,v in scalars.items():
        writer.add_scalar(prefix+k, v, step)
