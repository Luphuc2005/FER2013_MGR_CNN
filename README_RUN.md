# Cách Chạy Trên Máy Giảng Viên

Máy đã có sẵn môi trường. Không cần cài Anaconda, Python, CUDA hay tạo môi trường mới.

Đây là bản TensorFlow/Keras port theo file C-relation, kết quả gốc PyTorch có TTA hflip khoảng 74.3383%.

## 1. Vào thư mục project

```bash
cd /path/to/FER2013_SGU
```

## 2. Cấp quyền chạy script

```bash
chmod +x run_train.sh run_train_single_gpu.sh run_eval.sh
```

## 3. Chạy training 2 GPU

```bash
bash run_train.sh
```

Neu data/mask nam ngoai thu muc project, truyen path truc tiep:

```bash
MGR_DATA_PATH="/duong/dan/fer13-split" \
MGR_MASK_DIR="/duong/dan/mediapipe_region_masks" \
bash run_train.sh
```

Script sẽ:

- kiểm tra TensorFlow và GPU
- dùng GPU 0 và GPU 1
- profile mặc định ưu tiên ổn định RAM: 16 thread tính toán, 4 luồng `tf.data`
- batch mặc định 8/GPU, global batch 16
- nếu hết VRAM thì tự retry 4/GPU
- train full dataset, không giới hạn số mẫu
- giảm `shuffle_buffer`, `prefetch` và số luồng đọc dữ liệu để tránh RAM tăng dần
- không preload pixel/mask mặc định để tránh spike RAM cuối đầu run
- mặc định không bật `numactl` để tránh lỗi quyền trên server; có thể bật thủ công sau khi chạy ổn định
- lưu log vào `logs/`
- lưu checkpoint vào `outputs/tf_runs/c_relation_tokens_080_020_tf/checkpoints/`
- `best/` là checkpoint theo validation accuracy tốt nhất, `last/` dùng để resume, `periodic/` lưu mỗi 10 epoch

Mặc định script set:

```bash
TF_NUM_INTRAOP_THREADS=16
TF_NUM_INTEROP_THREADS=4
MGR_TF_DATA_NUM_PARALLEL_CALLS=4
MGR_TF_DATA_PRIVATE_THREADPOOL_SIZE=4
MGR_PREFETCH_BUFFER=1
MGR_SHUFFLE_BUFFER=512
MGR_PREDECODE_PIXELS=0
MGR_PRELOAD_MASKS=0
MGR_CACHE_DATA=0
```

Nếu máy bị lag khi đang dùng việc khác, có thể giảm thread trước khi chạy:

```bash
MGR_TF_DATA_NUM_PARALLEL_CALLS=2 MGR_TF_DATA_PRIVATE_THREADPOOL_SIZE=2 MGR_PRIMARY_BATCH_SIZE_PER_GPU=1 bash run_train.sh
```

Nếu đã chạy ổn định nhiều epoch và RAM không tăng bất thường, có thể thử profile nhanh hơn:

```bash
MGR_PRIMARY_BATCH_SIZE_PER_GPU=4 MGR_TF_DATA_NUM_PARALLEL_CALLS=8 MGR_TF_DATA_PRIVATE_THREADPOOL_SIZE=8 MGR_PREFETCH_BUFFER=1 bash run_train.sh
```

Neu muon ép đúng batch 16/GPU theo file C gốc, chỉ thử sau khi batch 2/GPU đã ổn định:

```bash
MGR_PRIMARY_BATCH_SIZE_PER_GPU=16 MGR_FALLBACK_BATCH_SIZE_PER_GPU=8 bash run_train.sh
```

Chỉ bật NUMA interleave khi server cho phép:

```bash
MGR_USE_NUMACTL=1 bash run_train.sh
```

## 4. Chạy 1 GPU nếu cần

```bash
bash run_train_single_gpu.sh
```

File này là bản debug/nhẹ nhất: 1 GPU, batch 1/GPU, CPU thread nhẹ hơn.

Nếu test trên Windows PowerShell với máy 1 GPU, dùng file riêng:

```powershell
conda activate mgr_tf210
cd "D:\HocTap\Phân tích  và xử lý ảnh\sgu-2026-facial-expression-recognition\FER2013_SGU"
powershell -ExecutionPolicy Bypass -File .\run_train_local_1gpu.ps1 -Epochs 50 -BatchSizePerGpu 1
```

File PowerShell này tự tắt yêu cầu 2 GPU, giảm `tf.data` xuống 1 luồng, `prefetch=1` và mặc định chỉ lấy `512` mẫu train, `128` mẫu validation/test để test nhanh trên máy local.

Nếu muốn test lâu hơn nhưng vẫn nhẹ:

```powershell
.\run_train_local_1gpu.ps1 -Epochs 50 -BatchSizePerGpu 1 -MaxTrainSamples 2048 -MaxValSamples 512 -MaxTestSamples 512
```

Nếu muốn train full dataset trên máy local 1 GPU, đặt giới hạn bằng `0`, nhưng máy có thể rất chậm và dễ đầy RAM/VRAM:

```powershell
.\run_train_local_1gpu.ps1 -Epochs 50 -BatchSizePerGpu 1 -MaxTrainSamples 0 -MaxValSamples 0 -MaxTestSamples 0
```

Bản full nên chạy trên server bằng `bash run_train.sh`.

## 5. Đánh giá checkpoint

```bash
bash run_eval.sh test
```

Hoặc:

```bash
bash run_eval.sh val
```

## 6. Nếu thiếu thư viện

```bash
python3 check_environment.py
```

Nếu có package thiếu, xem `MISSING_PACKAGES.txt`. Project không tự động `pip install`.
