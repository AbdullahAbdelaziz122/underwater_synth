import argparse, json, yaml
from pathlib import Path
from pipeline import generate
import os
def main():
    # Default arguments
    config = os.path.join(Path(__file__).parent, "./configs/Enhanced.yaml")
    out = os.path.join(Path(__file__).parent, "./data/generated_samples")
    
    
    ap = argparse.ArgumentParser(
        description="Generate physics-based underwater acoustic signals"
    )
    ap.add_argument("--config", type=str, default= config,
                    help="Path to YAML configuration file")
    ap.add_argument("--out", type=str, default= out,
                    help="Output directory for generated samples")
    ap.add_argument("--n", type=int, default=20,
                    help="Number of samples to generate")
    ap.add_argument("--class_name", type=str, default=None, 
                    choices=[None, "submarine", "torpedo"],
                    help="Generate only this class (default: alternate)")
    ap.add_argument("--seed", type=int, default=1337,
                    help="Random seed for reproducibility")
    ap.add_argument("--bellhop_mode", action="store_true",
                    help="Generate clean source signals for Bellhop Stage 2 (no propagation/noise)")
    args = ap.parse_args()

    # Load configuration
    if args.config is None:
        cfg = yaml.safe_load(open(Path(__file__).parent / "default_config.yaml", "r"))
    else:
        cfg = yaml.safe_load(open(args.config, "r"))
        
    
    
    # Generate samples
    metas = generate(cfg, args.out, args.n, 
                    class_filter=args.class_name, 
                    seed=args.seed,
                    bellhop_mode=args.bellhop_mode)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Generated {len(metas)} samples to {args.out}")
    print(f"Mode: {'Bellhop-ready (clean sources)' if args.bellhop_mode else 'Stage 1 validation (with noise)'}")
    print(f"{'='*60}\n")
    
    # Show first 3 samples
    print("First 3 metadata entries:")
    for i, m in enumerate(metas[:3], 1):
        print(f"\n--- Sample {i} ---")
        print(json.dumps(m, indent=2))
    
    # Class distribution
    classes = [m['class'] for m in metas]
    print(f"\n{'='*60}")
    print(f"Class distribution:")
    print(f"  Submarine: {classes.count('submarine')}")
    print(f"  Torpedo: {classes.count('torpedo')}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
