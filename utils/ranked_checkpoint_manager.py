from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional


class RankedCheckpointManager:
    """Keep the best K TensorFlow checkpoints ranked by an external metric.

    Unlike ``tf.train.CheckpointManager(max_to_keep=K)``, retention is based on
    metric rank rather than save time. Checkpoints are written with
    ``Checkpoint.write`` and the worst-ranked prefix is deleted after a new
    candidate enters the table.
    """

    _CKPT_RE = re.compile(r"^ckpt-(\d+)$")

    def __init__(
        self,
        checkpoint,
        directory: Path,
        max_to_keep: int,
        metric_name: str,
        mode: str,
        history_csv: Optional[Path] = None,
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.checkpoint = checkpoint
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max(int(max_to_keep), 0)
        self.metric_name = str(metric_name)
        self.mode = mode
        self.metadata_path = self.directory / "top_k_rankings.json"
        self.entries = self._load_entries(Path(history_csv) if history_csv else None)
        self._sort_entries()
        self._trim_to_limit()
        self._write_metadata_and_state()

    @property
    def latest_checkpoint(self) -> Optional[str]:
        """Return rank 1 for compatibility with final restore/rollback code."""
        if not self.entries:
            return None
        return str(self.directory / self.entries[0]["checkpoint"])

    @property
    def threshold(self) -> Optional[float]:
        if len(self.entries) < self.max_to_keep or not self.entries:
            return None
        return float(self.entries[-1]["metric"])

    def consider(self, epoch: int, metric: float, metrics: Optional[Dict[str, float]] = None) -> Dict[str, object]:
        epoch = int(epoch)
        metric = float(metric)
        if self.max_to_keep <= 0:
            return {"saved": False, "reason": "disabled", "removed": None, "threshold": None}
        if not math.isfinite(metric):
            raise ValueError(f"Non-finite {self.metric_name} at epoch {epoch}: {metric}")

        checkpoint_name = f"ckpt-{epoch}"
        existing = next((entry for entry in self.entries if entry["checkpoint"] == checkpoint_name), None)
        comparison_entries = [entry for entry in self.entries if entry["checkpoint"] != checkpoint_name]
        threshold = None
        if len(comparison_entries) >= self.max_to_keep:
            threshold = float(self._sorted(comparison_entries)[self.max_to_keep - 1]["metric"])
            qualifies = metric > threshold if self.mode == "max" else metric < threshold
        else:
            qualifies = True

        if existing is not None and float(existing["metric"]) == metric:
            return {
                "saved": False,
                "reason": "epoch_already_ranked",
                "removed": None,
                "threshold": threshold,
            }
        if not qualifies:
            return {
                "saved": False,
                "reason": "not_better_than_rank_k",
                "removed": None,
                "threshold": threshold,
            }

        prefix = self.directory / checkpoint_name
        self.checkpoint.write(str(prefix))

        entry = {
            "epoch": epoch,
            "checkpoint": checkpoint_name,
            "metric": metric,
        }
        for key, value in (metrics or {}).items():
            value = float(value)
            if math.isfinite(value):
                entry[str(key)] = value

        self.entries = comparison_entries + [entry]
        self._sort_entries()
        removed = None
        if len(self.entries) > self.max_to_keep:
            removed = self.entries.pop()
            self._delete_prefix(self.directory / removed["checkpoint"])
        self._write_metadata_and_state()
        return {
            "saved": True,
            "reason": "entered_top_k",
            "removed": removed,
            "threshold": threshold,
            "rank": next(i for i, item in enumerate(self.entries, 1) if item["checkpoint"] == checkpoint_name),
        }

    def _sorted(self, entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
        reverse = self.mode == "max"
        return sorted(entries, key=lambda item: (float(item["metric"]), int(item["epoch"])), reverse=reverse)

    def _sort_entries(self) -> None:
        self.entries = self._sorted(self.entries)

    def _trim_to_limit(self) -> None:
        if self.max_to_keep <= 0:
            return
        while len(self.entries) > self.max_to_keep:
            removed = self.entries.pop()
            self._delete_prefix(self.directory / removed["checkpoint"])

    def _delete_prefix(self, prefix: Path) -> None:
        if not self._CKPT_RE.fullmatch(prefix.name):
            raise ValueError(f"Refusing to delete unexpected checkpoint prefix: {prefix}")
        for path in self.directory.glob(prefix.name + ".*"):
            if path.is_file():
                path.unlink()

    def _load_entries(self, history_csv: Optional[Path]) -> List[Dict[str, object]]:
        index_names = {path.name[:-6] for path in self.directory.glob("ckpt-*.index")}
        if self.metadata_path.exists():
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if payload.get("metric_name") != self.metric_name or payload.get("mode") != self.mode:
                raise ValueError(f"Ranking metadata contract mismatch: {self.metadata_path}")
            entries = list(payload.get("checkpoints", []))
            ranked_names = {str(entry.get("checkpoint", "")) for entry in entries}
            missing_files = ranked_names - index_names
            unranked_files = index_names - ranked_names
            if missing_files or unranked_files:
                raise RuntimeError(
                    f"Checkpoint/metadata mismatch in {self.directory}: "
                    f"missing_files={sorted(missing_files)}, unranked_files={sorted(unranked_files)}"
                )
            return entries

        if not index_names:
            return []

        history_by_epoch: Dict[int, Dict[str, str]] = {}
        if history_csv and history_csv.exists():
            with history_csv.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("epoch"):
                        history_by_epoch[int(float(row["epoch"]))] = row

        entries: List[Dict[str, object]] = []
        unresolved = []
        for checkpoint_name in sorted(index_names):
            match = self._CKPT_RE.fullmatch(checkpoint_name)
            epoch = int(match.group(1)) if match else -1
            row = history_by_epoch.get(epoch, {})
            raw_metric = row.get(self.metric_name)
            if raw_metric in (None, ""):
                unresolved.append(checkpoint_name)
                continue
            entry: Dict[str, object] = {
                "epoch": epoch,
                "checkpoint": checkpoint_name,
                "metric": float(raw_metric),
            }
            for key in ("val_accuracy", "val_loss"):
                if row.get(key) not in (None, ""):
                    entry[key] = float(row[key])
            entries.append(entry)

        if unresolved:
            raise RuntimeError(
                f"Cannot reconstruct {self.metric_name} ranking for {unresolved} in {self.directory}. "
                f"Keep training_history.csv or provide top_k_rankings.json before resuming."
            )
        return entries

    def _write_metadata_and_state(self) -> None:
        payload = {
            "metric_name": self.metric_name,
            "mode": self.mode,
            "max_to_keep": self.max_to_keep,
            "checkpoints": [dict(entry, rank=rank) for rank, entry in enumerate(self.entries, 1)],
        }
        tmp_metadata = self.metadata_path.with_suffix(".json.tmp")
        tmp_metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_metadata.replace(self.metadata_path)

        state_path = self.directory / "checkpoint"
        if not self.entries or self.max_to_keep <= 0:
            if state_path.exists():
                state_path.unlink()
            return
        lines = [f'model_checkpoint_path: "{self.entries[0]["checkpoint"]}"']
        lines.extend(f'all_model_checkpoint_paths: "{entry["checkpoint"]}"' for entry in self.entries)
        tmp_state = self.directory / "checkpoint.tmp"
        tmp_state.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp_state.replace(state_path)
