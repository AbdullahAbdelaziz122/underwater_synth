import argparse, json, yaml
from pathlib import Path
from .pipeline import generate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--class_name", type=str, default=None, choices=[None, "submarine", "torpedo"])
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    metas = generate(cfg, args.out, args.n, class_filter=args.class_name, seed=args.seed)
    print(f"Generated {len(metas)} samples to {args.out}")
    print("First 3 metadata rows:")
    for m in metas[:3]:
        print(json.dumps(m, indent=2))

if __name__ == "__main__":
    main()
