#!/usr/bin/env python3
"""Interpola checkpoints compativeis e registra proveniencia."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha deve estar entre 0 e 1")

    base = torch.load(args.base, map_location="cpu", weights_only=True)
    adapted = torch.load(args.adapted, map_location="cpu", weights_only=True)
    if base.keys() != adapted.keys():
        raise ValueError("checkpoints possuem chaves diferentes")
    blended = {}
    for key in base:
        left, right = base[key], adapted[key]
        if left.shape != right.shape:
            raise ValueError(f"shape diferente em {key}")
        if torch.is_floating_point(left):
            blended[key] = left.mul(1.0 - args.alpha).add(right, alpha=args.alpha)
        else:
            blended[key] = right.clone()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blended, args.output)
    provenance = {
        "method": "linear_state_dict_interpolation",
        "base": str(args.base.resolve()),
        "base_sha256": digest(args.base),
        "adapted": str(args.adapted.resolve()),
        "adapted_sha256": digest(args.adapted),
        "alpha_adapted": args.alpha,
        "output": str(args.output.resolve()),
        "output_sha256": digest(args.output),
    }
    args.output.with_suffix(".blend.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
