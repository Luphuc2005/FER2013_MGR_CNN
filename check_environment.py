from __future__ import annotations

import importlib
import os
import platform
import sys
from typing import List, Tuple


REQUIRED_PACKAGES: List[Tuple[str, str, str]] = [
    ("tensorflow", "2.10.1", "tensorflow==2.10.1"),
    ("tensorflow_addons", "0.20.0", "tensorflow-addons==0.20.0"),
    ("numpy", "1.23.5", "numpy==1.23.5"),
    ("pandas", "1.5.3", "pandas==1.5.3"),
    ("sklearn", "1.3.2", "scikit-learn==1.3.2"),
    ("yaml", "6.0.2", "PyYAML==6.0.2"),
    ("PIL", "10.4.0", "Pillow==10.4.0"),
]


def check_imports() -> List[str]:
    missing = []
    for module_name, version, pip_spec in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(module_name)
            found = getattr(module, "__version__", "unknown")
            print(f"[OK] {module_name}: {found}")
        except Exception as exc:
            print(f"[MISSING] {module_name}: {exc}")
            missing.append(f"{module_name} | recommended: {version} | pip install {pip_spec}")
    return missing


def check_tensorflow() -> int:
    import tensorflow as tf

    print("TensorFlow:", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    gpus = tf.config.list_physical_devices("GPU")
    print("GPU count:", len(gpus))
    for idx, gpu in enumerate(gpus):
        print(f"GPU {idx}: {gpu.name}")
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print(f"  memory_growth warning: {exc}")
    for idx in range(len(gpus)):
        with tf.device(f"/GPU:{idx}"):
            x = tf.random.uniform([256, 256])
            y = tf.matmul(x, x)
            print(f"[OK] Tensor test on GPU:{idx}: shape={y.shape}, mean={float(tf.reduce_mean(y).numpy()):.6f}")
    if not gpus:
        print("[WARN] TensorFlow does not see any GPU. Training will fail if config requires GPUs.")
    return len(gpus)


def main() -> int:
    print("OS:", platform.platform())
    print("Python:", sys.version.replace("\n", " "))
    missing = check_imports()
    if missing:
        with open("MISSING_PACKAGES.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(missing) + "\n")
        print("[WARN] Missing packages were written to MISSING_PACKAGES.txt")
        return 1
    gpu_count = check_tensorflow()
    if gpu_count < 1 and os.environ.get("MGR_ALLOW_CPU", "").strip().lower() in {"1", "true", "yes", "on"}:
        print("[INFO] MGR_ALLOW_CPU is enabled; continuing without a visible GPU.")
        return 0
    return 0 if gpu_count >= 1 else 2


if __name__ == "__main__":
    raise SystemExit(main())
