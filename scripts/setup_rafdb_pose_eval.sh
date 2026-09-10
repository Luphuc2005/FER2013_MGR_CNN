#!/bin/bash
# Run once on the HPC login node. Separate OpenCV environment protects TensorFlow dependencies.
set -euo pipefail
ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
mkdir -p logs pretrained/pose_eval
if [[ ! -x pose_eval_env/bin/python ]]; then
    "$ROOT/fer2013_env/bin/python" -m venv "$ROOT/pose_eval_env"
fi
pose_eval_env/bin/python -m pip install 'numpy==1.26.4' 'Pillow==10.4.0' 'opencv-python-headless==4.10.0.84'
MODEL="$ROOT/pretrained/pose_eval/face_detection_yunet_2023mar.onnx"
if [[ ! -f "$MODEL" ]]; then
    curl --fail --location --retry 3 \
      'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx' \
      --output "$MODEL.download"
    mv "$MODEL.download" "$MODEL"
fi
pose_eval_env/bin/python -c "import cv2,numpy as np; d=cv2.FaceDetectorYN.create('$MODEL','',(320,320)); d.detect(np.zeros((320,320,3),np.uint8)); print('YUNET_ENV_OK',cv2.__version__)"
