# Hướng dẫn Chạy Pipeline FER2013 SigLIP2 trên Máy Server Thầy (Titan Z / Dual GPU)

Thư mục này chứa file cấu hình và script tự động hóa end-to-end cho mô hình **ConvNeXt-Base MS1M Adaptive SigLIP2 + Confusion-Aware Hard Semantic Separation**.

---

## 📁 Cấu trúc Thư mục `run_paper/`

- **`config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml`**: Configuration chuẩn dựa trên `run_convnext_base_ms1m_adaptive_siglip2_confusion_v100.slurm.sh`.
- **`run_siglip2_confusion_titan_z.sh`**: Script Bash tự động chạy full 3 bước:
  1. **Huấn luyện mô hình** (Auto-increment `output_dir` sang `_v2`, `_v3`... nếu đã có checkpoint cũ).
  2. **Quét trọng số TTA (Grid Search)** trên `best_accuracy` và `best_loss` checkpoints.
  3. **Đánh giá Top-5 Checkpoint Softmax Ensemble + TTA**.

---

## 🚀 Cách chạy trên Máy Server (Titan Z / 2 GPUs)

### Cách 1: Chạy trực tiếp trên Terminal

```bash
# 1. Cấp quyền thực thi cho script (chỉ cần làm 1 lần)
chmod +x run_paper/run_siglip2_confusion_titan_z.sh

# 2. Chạy script
bash run_paper/run_siglip2_confusion_titan_z.sh
```

---

### Cách 2: Chạy ẩn dưới nền (Nohup / Background)
Dành cho trường hợp chạy lâu và muốn ngắt SSH mà server vẫn tiếp tục chạy:

```bash
nohup bash run_paper/run_siglip2_confusion_titan_z.sh > logs/nohup_siglip2.log 2>&1 &
```

Để theo dõi tiến độ log realtime:
```bash
tail -f logs/run_paper_siglip2_confusion_*.log
```

---

## ⚙️ Tùy chỉnh GPU (Nếu muốn)

Mặc định script sử dụng cả 2 GPU (`CUDA_VISIBLE_DEVICES=0,1`). Nếu bạn muốn chỉ dùng 1 GPU (ví dụ GPU 0):

```bash
CUDA_VISIBLE_DEVICES=0 bash run_paper/run_siglip2_confusion_titan_z.sh
```
