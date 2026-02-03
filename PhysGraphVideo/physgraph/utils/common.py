import yaml, os
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    if "_base_" in base:
        base_path = base["_base_"]
        if not os.path.isabs(base_path):
            base_dir = os.path.dirname(path)
            base_path = os.path.join(base_dir, os.path.basename(base_path))
        parent = load_config(base_path)
        for k,v in base.items():
            if k == "_base_": continue
            parent[k] = v
        return parent
    return base
class NS:
    def __init__(self, d):
        for k,v in d.items():
            setattr(self, k, NS(v) if isinstance(v, dict) else v)
def to_ns(d): return NS(d)
