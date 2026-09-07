"""Offline regression checks; synthetic geometry, no RAF data or TensorFlow required."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from estimate_rafdb_head_pose import OBJECT_POINTS, LOCAL, estimate, solve_pose
from evaluate_rafdb_pose import summarize
from rafdb_pose_common import file_hash, pose_group, read_test


class PoseTests(unittest.TestCase):
    def test_recover_known_camera_rotations(self):
        camera = np.array([[320., 0, 160], [0, 320, 160], [0, 0, 1]])
        for yaw, pitch, roll in [(0, 0, 0), (35, 0, 0), (-50, 0, 0),
                                  (0, 35, 0), (0, -50, 0), (0, 0, 45), (40, -20, 15)]:
            x, y, z = np.radians([pitch, yaw, roll])
            rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
            ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
            rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
            rvec, _ = cv2.Rodrigues(rz@ry@rx)
            image_points, _ = cv2.projectPoints(OBJECT_POINTS, rvec, np.array([0., 0, 500.]), camera, None)
            result = solve_pose(image_points, 320, 320)
            np.testing.assert_allclose([result['yaw'], result['pitch'], result['roll']], [yaw, pitch, roll], atol=1e-5)
            self.assertAlmostEqual(result['pose_angle'], max(abs(yaw), abs(pitch)), places=5)

    def test_thresholds(self):
        self.assertEqual([pose_group(x) for x in [0, 29.999, 30, 44.999, 45, 80]],
                         ['lt30', 'lt30', '30_to_lt45', '30_to_lt45', 'ge45', 'ge45'])

    def test_readonly_csv_label_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with (root/'test.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_path', 'label'])
                for i in range(1, 8):
                    path = root/f'test_{i}.png'
                    Image.new('RGB', (100, 100)).save(path)
                    writer.writerow([path.name, i])
            before = file_hash(root/'test.csv')
            rows = read_test(root, expected=7)
            self.assertEqual([r['label'] for r in rows], [5, 2, 1, 3, 4, 0, 6])
            self.assertEqual(before, file_hash(root/'test.csv'))
            with self.assertRaises(ValueError):
                read_test(root, expected=3068)

    def test_undetected_retained_and_not_assigned_frontal(self):
        class EmptyDetector:
            def setInputSize(self, size):
                pass
            def detect(self, image):
                return None, None
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'blank.png'
            Image.new('RGB', (100, 100)).save(path)
            LOCAL.detector = EmptyDetector()
            args = SimpleNamespace(detector_size=320)
            try:
                result = estimate(dict(image_path=str(path), label=0, sample_id=0), args)
            finally:
                del LOCAL.detector
            self.assertEqual(result['subset'], 'undetected')
            self.assertEqual(result['yaw'], '')
            self.assertEqual(result['face_detected'], 0)

    def test_metrics_include_failures_and_empty_groups(self):
        labels = np.array([0, 0, 1])
        probabilities = np.eye(7)[[0, 1, 1]]
        rows = [dict(face_detected=1, pose_valid=1), dict(face_detected=0, pose_valid=0),
                dict(face_detected=1, pose_valid=1)]
        metrics = summarize(labels, probabilities, rows, np.ones(3, bool), np.zeros(7))
        self.assertEqual(metrics['N'], 3)
        self.assertAlmostEqual(metrics['accuracy'], 2/3)
        self.assertAlmostEqual(metrics['detection_rate'], 2/3)
        self.assertAlmostEqual(metrics['macro_f1'], (2/3+2/3)/7)
        self.assertEqual(metrics['per_class_accuracy']['angry'], .5)
        self.assertIsNone(metrics['per_class_accuracy']['fear'])
        empty = summarize(labels, probabilities, rows, np.zeros(3, bool), np.zeros(7))
        self.assertEqual(empty['N'], 0)
        self.assertIsNone(empty['accuracy'])
        json.dumps(empty, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
