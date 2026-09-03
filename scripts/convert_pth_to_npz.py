#!/usr/bin/env python3
"""
Utility script to convert PyTorch .pth checkpoint to NumPy .npz file.
This allows running the TensorFlow FER pipeline on servers without PyTorch installed.
"""

from pathlib import Path
import sys

def main():
    pth_path = Path("pretrained/convnext_base_ms1m_arcface.pth")
    npz_path = Path("pretrained/convnext_base_ms1m_arcface.npz")
    
    if len(sys.argv) > 1:
        pth_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        npz_path = Path(sys.argv[2])
        
    if not pth_path.exists():
        print(f"[ERROR] Source PyTorch file not found: {pth_path}")
        return 1
        
    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch is required locally to run this conversion script.")
        return 1
        
    print(f"[INFO] Loading PyTorch checkpoint: {pth_path} ...")
    checkpoint = torch.load(str(pth_path), map_location="cpu")
    if isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    else:
        state = checkpoint
        
    np_dict = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            np_dict[k] = v.cpu().numpy()
            
    print(f"[INFO] Saving {len(np_dict)} tensor variables to NumPy file: {npz_path} ...")
    np.savez_compressed(str(npz_path), **np_dict)
    print(f"[SUCCESS] Conversion complete! File saved to: {npz_path} ({npz_path.stat().st_size / (1024*1024):.2f} MB)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
