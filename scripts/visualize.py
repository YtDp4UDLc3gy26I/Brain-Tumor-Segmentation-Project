#!/usr/bin/env python
import argparse
import torch
from src.viz.predict import load_model, showPredictsById

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to checkpoint .pt")
    p.add_argument("--root", required=True, help="BraTS root dir containing case folders")
    p.add_argument("--case", required=True, help="Case id, e.g., BraTS20_Training_017")
    p.add_argument("--slice", type=int, default=60)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--slices", type=int, default=100)
    p.add_argument("--start_at", type=int, default=22)
    args = p.parse_args()

    model = load_model(args.weights, in_ch=2, n_classes=4, dropout=0.35)
    showPredictsById(
        case=args.case, root_dir=args.root, model=model,
        img_size=args.img_size, volume_slices=args.slices, volume_start_at=args.start_at,
        start_slice=args.slice
    )

if __name__ == "__main__":
    cli()