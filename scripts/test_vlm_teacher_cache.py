"""CPU-only cache corruption and sample alignment regressions (no TensorFlow)."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from utils.vlm_teacher_cache import CLASSES, RECIPE, load_teacher_logits, source_hashes


class TeacherCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.images = [root / "a.jpg", root / "b.jpg"]
        for i, path in enumerate(self.images):
            path.write_bytes(bytes([i, 2, 3]))
        self.records = SimpleNamespace(images=np.array([str(p) for p in self.images]),
                                       sample_ids=np.array([0, 1]), labels=np.array([1, 4]))
        self.path = root / "teacher.npz"
        self.cfg = dict(vlm_teacher=dict(cache_path=str(self.path), model_name="siglip-test"))
        self.logits = np.arange(14, dtype=np.float32).reshape(2, 7)
        self.meta = dict(recipe=RECIPE, class_names=CLASSES, split="train",
                         model_name="siglip-test", prototype_sha256="verified-prototype",
                         model_commit="pinned", processor={"size": 224},
                         logits_sha256=hashlib.sha256(self.logits.tobytes()).hexdigest())
        self.save()
        self.patch = patch("utils.vlm_teacher_cache.prototype_contract", return_value="verified-prototype")
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def save(self):
        np.savez(self.path, metadata=json.dumps(self.meta), logits=self.logits,
                 images=self.records.images, sample_ids=self.records.sample_ids,
                 labels=self.records.labels, source_sha256=source_hashes(self.records.images))

    def test_valid_and_reordered_samples(self):
        np.testing.assert_array_equal(load_teacher_logits(self.cfg, self.records), self.logits)
        reversed_records = SimpleNamespace(**{k: v[::-1] for k, v in vars(self.records).items()})
        with self.assertRaisesRegex(ValueError, "alignment"):
            load_teacher_logits(self.cfg, reversed_records)

    def test_source_image_changed(self):
        self.images[0].write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            load_teacher_logits(self.cfg, self.records)

    def test_wrong_class_order(self):
        self.meta["class_names"] = CLASSES[::-1]
        self.save()
        with self.assertRaisesRegex(ValueError, "class_names"):
            load_teacher_logits(self.cfg, self.records)

    def test_corrupted_logits(self):
        self.logits[0, 0] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_teacher_logits(self.cfg, self.records)

    def test_test_targets_rejected(self):
        with self.assertRaisesRegex(ValueError, "training-only"):
            load_teacher_logits(self.cfg, self.records, "test")


if __name__ == "__main__":
    unittest.main()
