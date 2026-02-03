import os, json, argparse, subprocess, sys
def run(cmd):
    print("[RUN]", " ".join(cmd), flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True)
    print(out.stdout)
    if out.returncode != 0:
        print(out.stderr); raise SystemExit(out.returncode)
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run([sys.executable, "-m", "tools.train", "--config", "configs/cvpr_base.yaml", "--device", args.device])
    run([sys.executable, "-m", "tools.train", "--config", "configs/ablation_physics.yaml", "--device", args.device])
    run([sys.executable, "-m", "tools.train", "--config", "configs/ablation_graphrag.yaml", "--device", args.device])
    run([sys.executable, "-m", "tools.train", "--config", "configs/phys_plus_rag.yaml", "--device", args.device])
    summary = {
        "base": "runs/cvpr_base/best.pt",
        "physics": "runs/ablation_physics/best.pt",
        "graphrag": "runs/ablation_graphrag/best.pt",
        "phys_plus_rag": "runs/phys_plus_rag/best.pt"
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(summary)
if __name__ == "__main__": main()
