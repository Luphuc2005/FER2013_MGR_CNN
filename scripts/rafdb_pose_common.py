"""Read-only RAF test discovery and shared pose-evaluation utilities (no TensorFlow)."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
RAF_TO_MODEL = {1: 5, 2: 2, 3: 1, 4: 3, 5: 4, 6: 0, 7: 6}
GROUPS = ['lt30', '30_to_lt45', 'ge45', 'undetected']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows, fields=None):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pose_group(angle):
    return 'lt30' if angle < 30 else ('30_to_lt45' if angle < 45 else 'ge45')


def resolve_image(value, data_root, manifest_parent):
    p = Path(value)
    candidates = [p] if p.is_absolute() else [PROJECT_ROOT / p, manifest_parent / p, data_root / p]
    matches = {q.resolve() for q in candidates if q.is_file()}
    if len(matches) != 1:
        raise ValueError(f'Image path missing or ambiguous: {value}, candidates={matches}')
    return str(matches.pop())


def read_test(data_root, test_csv=None, label_space='auto', expected=3068):
    """Preserve CSV row order; map RAF raw labels once, without writing source files."""
    root = Path(data_root).resolve()
    manifest = Path(test_csv).resolve() if test_csv else root / 'test.csv'
    if not manifest.is_file() and not test_csv:
        candidates = sorted(root.rglob('test.csv'))
        if len(candidates) == 1:
            manifest = candidates[0]
        elif len(candidates) > 1:
            raise ValueError('Multiple test.csv files; supply --test-csv explicitly.')
    rows = []
    if manifest.is_file():
        with manifest.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            keys = reader.fieldnames or []
            path_key = next((k for k in ['image_path', 'filepath', 'path', 'image', 'file'] if k in keys), None)
            label_key = next((k for k in ['emotion', 'label', 'target', 'class', 'y'] if k in keys), None)
            if not path_key or not label_key:
                raise ValueError(f'{manifest} must have image paths and integer labels.')
            for i, row in enumerate(reader):
                rows.append(dict(sample_id=i, image_path=resolve_image(row[path_key], root, manifest.parent),
                                 label=int(row[label_key])))
        space = label_space
        unique = {r['label'] for r in rows}
        if space == 'auto':
            if 0 in unique and 7 not in unique:
                space = 'model'
            elif 7 in unique and 0 not in unique:
                space = 'raf1'
            else:
                raise ValueError('Ambiguous CSV label space; set --label-space model or raf1.')
        if space == 'raf1':
            for row in rows:
                row['label'] = RAF_TO_MODEL[row['label']]
    else:
        if test_csv:
            raise FileNotFoundError(manifest)
        test_dir = root / 'test'
        if not test_dir.is_dir():
            raise FileNotFoundError(f'Expected test.csv or {test_dir}; pass the current RAF test data root.')
        folders = sorted(p for p in test_dir.iterdir() if p.is_dir())
        prefixes = {p.name.lower().split('_')[0] for p in folders}
        raw = label_space == 'raf1' or (label_space == 'auto' and prefixes == set('1234567'))
        aliases = {name: i for i, name in enumerate(CLASSES)}
        aliases.update(anger=0, happiness=3, sadness=4)
        for folder in folders:
            key = folder.name.lower().split('_')[0]
            label = RAF_TO_MODEL[int(key)] if raw else (int(key) if key.isdigit() else aliases[key])
            for p in sorted(folder.rglob('*')):
                if p.is_file() and p.suffix.lower() in EXTENSIONS:
                    rows.append(dict(sample_id=len(rows), image_path=str(p.resolve()), label=label))
    if not rows or (expected and len(rows) != expected):
        raise ValueError(f'RAF full test count={len(rows)}, expected={expected}. No samples were silently dropped.')
    if any(r['label'] not in range(7) for r in rows):
        raise ValueError('Labels must map to the seven project classes.')
    if len({r['image_path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate test image paths.')
    return rows
