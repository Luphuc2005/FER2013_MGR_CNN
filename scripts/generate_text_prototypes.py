#!/usr/bin/env python3
"""
Pre-compute text prototype embeddings for CLIP / SigLIP / SigLIP2 models on Login Node (with internet).
Usage:
    python scripts/generate_text_prototypes.py --model google/siglip2-base-patch16-224 --dim 768 --multi
    python scripts/generate_text_prototypes.py --model google/siglip-base-patch16-224 --dim 768 --multi
    python scripts/generate_text_prototypes.py --model openai/clip-vit-base-patch32 --dim 512 --multi
"""

import argparse
import sys
import os

# Add root path to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    parser = argparse.ArgumentParser(description="Pre-compute text prototypes for CLIP / SigLIP models.")
    parser.add_argument("--model", type=str, default="google/siglip2-base-patch16-224", help="HuggingFace model repo name")
    parser.add_argument("--dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--multi", action="store_true", default=True, help="Generate 5-granularity prototypes")
    args = parser.parse_args()

    # Import inside main after path setup
    from models.clip_text_encoder import get_or_compute_clip_text_prototypes

    print(f"[PRE-COMPUTE] Generating prototypes for {args.model} (dim={args.dim}, multi={args.multi})...")
    protos = get_or_compute_clip_text_prototypes(
        model_name=args.model,
        embedding_dim=args.dim,
        multi_prototype=args.multi,
    )
    print(f"[SUCCESS] Generated prototypes array with shape {protos.shape}!")

if __name__ == "__main__":
    main()
