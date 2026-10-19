"""Threshold-independent, point-wise, event-wise, and attribution metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from faster_etapr import evaluate_from_preds
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

from .attacks import FILE_ATTACKS, targets
from .data import Stream

Array = NDArray[Any]


@dataclass
class ScoreRecord:
    stream: Stream
    ends: Array
    scores: Array
    sensor_scores: Array
    predictions: Array

    @property
    def labels(self) -> Array:
        return self.stream.y[self.ends].astype(np.uint8)


def binary_events(labels: Array) -> list[tuple[int, int]]:
    """Inclusive events from a binary point stream."""
    labels = np.asarray(labels, dtype=np.uint8)
    starts = np.flatnonzero(np.diff(np.concatenate(([0], labels))) == 1)
    ends = np.flatnonzero(np.diff(np.concatenate((labels, [0]))) == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _with_file_boundaries(arrays: list[Array]) -> Array:
    separated = []
    for array in arrays:
        separated.extend((np.asarray(array), np.zeros(1, dtype=np.uint8)))
    return np.concatenate(separated)


def per_attack_rows(
    records: list[ScoreRecord],
    tags: list[str],
    top_k: int,
    model: str,
    seed: int,
    arm: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        stream = record.stream
        attack_ids = FILE_ATTACKS[stream.version][stream.name]
        events = binary_events(stream.y)
        if len(events) != len(attack_ids):
            raise ValueError(
                f"Attack metadata mismatch for {stream.version}/{stream.name}: "
                f"{len(events)} observed vs {len(attack_ids)} documented"
            )
        for attack_id, (start, stop) in zip(attack_ids, events, strict=True):
            positions = np.flatnonzero((record.ends >= start) & (record.ends <= stop))
            detected = bool(len(positions) and record.predictions[positions].any())
            delay = np.nan
            if detected:
                first = positions[np.flatnonzero(record.predictions[positions])[0]]
                delay = float(stream.timestamps[record.ends[first]] - stream.timestamps[start])
            documented = targets(stream.version, attack_id)
            retained = sorted(set(documented) & set(tags))
            if len(positions):
                attribution = record.sensor_scores[positions].mean(axis=0)
                order = np.argsort(-attribution, kind="stable")
                ranked = [tags[index] for index in order]
            else:
                ranked = []
            ranks = [ranked.index(tag) + 1 for tag in retained if tag in ranked]
            best_rank = min(ranks) if ranks else np.nan
            hit = bool(ranks and best_rank <= top_k)
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "domain": stream.version,
                    "arm": arm,
                    "version": stream.version,
                    "file": stream.name,
                    "attack_id": attack_id,
                    "event_start": int(stream.timestamps[start]),
                    "event_end": int(stream.timestamps[stop]),
                    "duration_seconds": int(stream.timestamps[stop] - stream.timestamps[start] + 1),
                    "detected": detected,
                    "delay_seconds": delay,
                    "documented_targets": "|".join(documented),
                    "retained_targets": "|".join(retained),
                    "target_retained": bool(retained),
                    "best_target_rank": best_rank,
                    "attribution_hit": hit if retained else np.nan,
                    "top_sensors": "|".join(ranked[:top_k]),
                }
            )
    return rows


def evaluate_records(
    records: list[ScoreRecord],
    eval_stride: int,
    attack_rows: list[dict[str, Any]],
) -> dict[str, float | int]:
    if not records:
        raise ValueError("At least one score record is required")
    scores = np.concatenate([record.scores for record in records])
    labels = np.concatenate([record.labels for record in records])
    predictions = np.concatenate([record.predictions for record in records]).astype(np.uint8)
    labels_eta = _with_file_boundaries([record.labels for record in records])
    predictions_eta = _with_file_boundaries([record.predictions for record in records])
    all_metrics = evaluate_from_preds(predictions_eta, labels_eta, theta_p=0.5, theta_r=0.1)
    false_alarms = int(sum(np.sum(record.predictions[record.labels == 0]) for record in records))
    alarms = int(predictions.sum())
    duration_days = float(
        sum(
            (record.stream.timestamps[record.ends[-1]] - record.stream.timestamps[record.ends[0]] + eval_stride)
            / 86400.0
            for record in records
        )
    )
    delays = np.asarray([row["delay_seconds"] for row in attack_rows], dtype=float)
    detected = np.asarray([row["detected"] for row in attack_rows], dtype=bool)
    retained = np.asarray([row["target_retained"] for row in attack_rows], dtype=bool)
    hits = np.asarray([bool(row["attribution_hit"]) for row in attack_rows if row["target_retained"]])
    point_tp = int(np.sum((predictions == 1) & (labels == 1)))
    point_fp = int(np.sum((predictions == 1) & (labels == 0)))
    point_fn = int(np.sum((predictions == 0) & (labels == 1)))
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "point_precision": point_tp / max(point_tp + point_fp, 1),
        "point_recall": point_tp / max(point_tp + point_fn, 1),
        "point_f1": 2 * point_tp / max(2 * point_tp + point_fp + point_fn, 1),
        "eta_precision": float(all_metrics["eta/precision"]),
        "eta_recall": float(all_metrics["eta/recall"]),
        "eta_f1": float(all_metrics["eta/f1"]),
        "eta_detection_recall": float(all_metrics["eta/recall_detection"]),
        "alarms": alarms,
        "false_alarms": false_alarms,
        "false_alarms_per_day": false_alarms / duration_days,
        "duration_days": duration_days,
        "attack_count": len(attack_rows),
        "attacks_detected": int(detected.sum()),
        "attack_detection_rate": float(detected.mean()),
        "mean_delay_seconds": float(np.nanmean(delays)) if detected.any() else np.nan,
        "median_delay_seconds": float(np.nanmedian(delays)) if detected.any() else np.nan,
        "attribution_target_coverage": float(retained.mean()),
        "attribution_hit_at_k": float(hits.mean()) if len(hits) else np.nan,
    }
