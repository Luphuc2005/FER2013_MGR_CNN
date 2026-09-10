#!/usr/bin/env python3
"""Approximate head pose on current RAF crops: CPU YuNet + five-point PnP."""
from __future__ import annotations

import argparse
import json
import threading
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from rafdb_pose_common import CLASSES, GROUPS, PROJECT_ROOT, file_hash, pose_group, read_test, write_csv

YUNET_URL = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
# Generic facial template, arbitrary mm-like scale. Camera coordinates x right,
# y down, z away. Nose points towards camera (negative z); frontal R = identity.
# YuNet order: subject right eye, left eye, nose, right mouth, left mouth.
OBJECT_POINTS = np.array([[-32, -28, 0], [32, -28, 0], [0, 0, -28],
                          [-24, 28, -2], [24, 28, -2]], dtype=np.float64)
LOCAL = threading.local()


def solve_pose(points, width, height, focal_scale=1.0):
    focal = max(width, height) * focal_scale
    camera = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]], dtype=np.float64)
    points = np.asarray(points, dtype=np.float64).reshape(5, 2)
    ok, rvec, tvec = cv2.solvePnP(OBJECT_POINTS, points, camera, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise ValueError('pnp_failed')
    # ITERATIVE with supplied initial guess supports 5 correspondences.
    ok, rvec, tvec = cv2.solvePnP(OBJECT_POINTS, points, camera, None, rvec, tvec,
                                 True, flags=cv2.SOLVEPNP_ITERATIVE)
    rotation, _ = cv2.Rodrigues(rvec)
    if not ok or np.any((OBJECT_POINTS @ rotation.T + tvec.reshape(1, 3))[:, 2] <= 0):
        raise ValueError('pnp_behind_camera')
    # R = Rz(roll) Ry(yaw) Rx(pitch); no 180-degree folding/clamping.
    yaw = np.degrees(np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0])))
    pitch = np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))
    roll = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
    projected, _ = cv2.projectPoints(OBJECT_POINTS, rvec, tvec, camera, None)
    error = np.sqrt(np.mean(np.sum((projected.reshape(5, 2) - points)**2, axis=1)))
    norm_error = error / max(np.linalg.norm(points[0] - points[1]), 1.0)
    if not np.isfinite([yaw, pitch, roll, norm_error]).all():
        raise ValueError('pnp_nonfinite')
    return dict(yaw=float(yaw), pitch=float(pitch), roll=float(roll),
                pose_angle=float(max(abs(yaw), abs(pitch))), reprojection_error=float(norm_error))


def estimate(row, args):
    result = dict(row, yaw='', pitch='', roll='', pose_angle='', subset='undetected',
                  face_detected=0, pose_valid=0, detector_score='', n_faces=0,
                  reprojection_error='', status='no_face', landmarks='[]')
    # Read errors are fatal: never allow unreadable images to masquerade as no face.
    with Image.open(row['image_path']) as im:
        rgb = np.asarray(im.convert('RGB'))
    h, w = rgb.shape[:2]
    scale = args.detector_size / max(h, w)
    image = cv2.resize(rgb[:, :, ::-1], (round(w*scale), round(h*scale)))
    if not hasattr(LOCAL, 'detector'):
        LOCAL.detector = cv2.FaceDetectorYN.create(str(args.yunet), '', (320, 320), args.score_threshold,
                                                   0.3, 5000, cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
    LOCAL.detector.setInputSize((image.shape[1], image.shape[0]))
    _, faces = LOCAL.detector.detect(image)
    if faces is None or not len(faces):
        return result
    # RAF inputs are already face crops: select largest face, record ambiguity.
    face = max(faces, key=lambda f: f[2]*f[3])
    points = face[4:14].reshape(5, 2).astype(float)
    points[:, 0] *= w / image.shape[1]
    points[:, 1] *= h / image.shape[0]
    result.update(face_detected=1, detector_score=float(face[-1]), n_faces=len(faces),
                  landmarks=json.dumps(points.tolist()))
    try:
        pose = solve_pose(points, w, h, args.focal_scale)
        result.update(pose)
        if abs(pose['pitch']) > 90 or abs(pose['roll']) > 90:
            result['status'] = 'implausible_orientation'
        elif pose['reprojection_error'] > args.max_reprojection_error:
            result['status'] = 'high_reprojection_error'
        else:
            result.update(pose_valid=1, subset=pose_group(pose['pose_angle']), status='ok')
    except ValueError as exc:
        result['status'] = str(exc)
    except cv2.error:
        result['status'] = 'pnp_opencv_error'
    return result


def contact_sheets(rows, out, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for group in GROUPS:
        members = [r for r in rows if r['subset'] == group]
        chosen = rng.choice(len(members), min(16, len(members)), replace=False) if members else []
        sheet = Image.new('RGB', (4*224, 4*268), 'white')
        draw = ImageDraw.Draw(sheet)
        for j, index in enumerate(chosen):
            row = members[int(index)]
            selected.append(row)
            with Image.open(row['image_path']) as im:
                picture = im.convert('RGB')
            overlay = ImageDraw.Draw(picture)
            for x, y in json.loads(row['landmarks']):
                overlay.ellipse((x-1.5, y-1.5, x+1.5, y+1.5), fill='lime')
            picture = ImageOps.contain(picture, (224, 224))
            x, y = (j % 4)*224, (j//4)*268
            sheet.paste(picture, (x, y))
            angle = f"{row['pose_angle']:.1f}" if row['pose_angle'] != '' else 'NA'
            draw.text((x+3, y+225), f"{Path(row['image_path']).name}\n{CLASSES[row['label']]} pose={angle}\n{row['status']}", fill='black')
        sheet.save(out / f'inspect_{group}.jpg')
    if selected:
        write_csv(out / 'inspect_samples.csv', selected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=PROJECT_ROOT/'data/rafdb')
    p.add_argument('--test-csv', type=Path)
    p.add_argument('--label-space', choices=['auto', 'model', 'raf1'], default='auto')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--yunet', type=Path, default=PROJECT_ROOT/'pretrained/pose_eval/face_detection_yunet_2023mar.onnx')
    p.add_argument('--download-yunet', action='store_true')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--detector-size', type=int, default=320)
    p.add_argument('--score-threshold', type=float, default=0.7)
    p.add_argument('--focal-scale', type=float, default=1.0)
    p.add_argument('--max-reprojection-error', type=float, default=0.15)
    p.add_argument('--expected-total', type=int, default=3068)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.workers < 1 or args.focal_scale <= 0 or args.detector_size < 32:
        p.error('Invalid worker count, focal scale, or detector size.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {args.output_dir}')
    rows = read_test(args.data_root, args.test_csv, args.label_space, args.expected_total)
    if not args.yunet.exists():
        if not args.download_yunet:
            raise FileNotFoundError('YuNet missing; use --download-yunet once on a node with network access.')
        args.yunet.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(YUNET_URL, timeout=45) as response:
            payload = response.read()
        if len(payload) < 100000:
            raise ValueError('Invalid YuNet download (possibly an LFS pointer).')
        args.yunet.write_bytes(payload)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    # Fail early if ONNX/backend incompatible, rather than labelling every image undetected.
    cv2.FaceDetectorYN.create(str(args.yunet), '', (320, 320)).detect(np.zeros((320, 320, 3), np.uint8))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(lambda row: estimate(row, args), rows), 1):
            results.append(result)
            if i % 100 == 0 or i == len(rows):
                print(f'POSE_PROGRESS {i}/{len(rows)}', flush=True)
    write_csv(args.output_dir/'rafdb_test_pose.csv', results)
    contact_sheets(results, args.output_dir, args.seed)
    metadata = dict(method='YuNet five-landmark + generic-template SQPnP/iterative PnP',
                    protocol='estimated_pose_on_current_raf_test_crops_not_Wang_benchmark',
                    pose_angle='max(abs(yaw), abs(pitch))', euler='Rz(roll) Ry(yaw) Rx(pitch), degrees',
                    object_points=OBJECT_POINTS.tolist(), yunet_url=YUNET_URL,
                    yunet_sha256=file_hash(args.yunet), opencv_version=cv2.__version__,
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    N=len(results), detection_rate=sum(r['face_detected'] for r in results)/len(results),
                    pose_valid_rate=sum(r['pose_valid'] for r in results)/len(results),
                    groups=dict(Counter(r['subset'] for r in results)),
                    status_counts=dict(Counter(r['status'] for r in results)),
                    limitation='Uncalibrated focal length and generic 5-point face geometry; approximate angles on aligned crops. Detection failures can be pose-dependent.')
    (args.output_dir/'pose_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('RAFDB_POSE_ESTIMATION_OK ' + json.dumps(metadata['groups']), flush=True)


if __name__ == '__main__':
    main()
