# FER2013_MGR_CNN

TensorFlow MGR-CNN C-Relation training package for FER2013_SGU.

Ban nay chay bang TensorFlow/Keras, port theo cau hinh C-relation:

```text
configs/paper_ablation_strong/c_cnn_region_logits_080_020_imagenet_sam_cosine_seed42_relation_noclean_batch16_kaggle.yaml
```

Ket qua PyTorch goc da ghi nhan:

- no-TTA: 74.1711%
- TTA hflip: 74.3383%

Project khong tu cai Anaconda, Python, CUDA, cuDNN hay TensorFlow. May da co moi truong san thi chi can chay script.

## Chay Training

```bash
cd /path/to/FER2013_SGU
chmod +x run_train.sh run_train_single_gpu.sh run_eval.sh
bash run_train.sh
```

Neu chi muon dung 1 GPU:

```bash
bash run_train_single_gpu.sh
```

## Chay Kaggle 2 GPU

Gan 2 Kaggle Dataset input nay vao notebook:

- `/kaggle/input/datasets/doduyquynii/fer13-split`
- `/kaggle/input/datasets/lhongphuc2/mediapipe-mask-datasets-35887`

Chay:

```bash
chmod +x run_train_kaggle_2gpu.sh
bash run_train_kaggle_2gpu.sh
```

Mac dinh Kaggle launcher dung 2 GPU, batch `2/GPU`, full dataset, output vao `/kaggle/working/outputs/...`.
Neu muon thu batch lon hon:

```bash
MGR_PRIMARY_BATCH_SIZE_PER_GPU=4 bash run_train_kaggle_2gpu.sh
```

## Kiem Tra Moi Truong

```bash
python3 check_environment.py
```

Neu thieu thu vien, xem `MISSING_PACKAGES.txt`. Project khong tu dong cai thu vien.

## File Chinh

- `config.yaml`: cau hinh TensorFlow port theo C-relation
- `train.py`: training loop TensorFlow voi `MirroredStrategy`
- `evaluate.py`: danh gia checkpoint TensorFlow
- `check_environment.py`: kiem tra Python, TensorFlow, GPU va package
- `run_train.sh`: chay training 2 GPU
- `run_train_single_gpu.sh`: chay training 1 GPU
- `run_eval.sh`: danh gia val/test

## Output

```text
logs/
outputs/tf_runs/c_relation_tokens_080_020_tf/
  checkpoints/
    best/
    last/
    periodic/
  training_history.csv
  test_metrics.json
```
