"""HAI 21.03/23.05 loaders, common-tag alignment, and causal windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

Array = NDArray[Any]
FILES = {
    "21.03": {
        "train": [f"train{i}" for i in range(1, 4)],
        "test": [f"test{i}" for i in range(1, 6)],
    },
    "23.05": {"train": [], "test": ["test1", "test2"]},
}


@dataclass
class Stream:
    version: str
    name: str
    x: Array
    y: Array
    timestamps: Array
    train_end: int | None = None
    label_alignment: str = ""

    def __len__(self) -> int:
        return len(self.y)


def raw_path(raw_dir: Path, version: str, name: str) -> Path:
    if version == "21.03":
        return raw_dir / f"hai-{version}" / f"{name}.csv.gz"
    return raw_dir / f"hai-{version}" / f"hai-{name}.csv"


def _header(raw_dir: Path, version: str, name: str) -> list[str]:
    return pd.read_csv(raw_path(raw_dir, version, name), nrows=0).columns.tolist()


def common_tags(raw_dir: Path) -> list[str]:
    old = _header(raw_dir, "21.03", "train1")
    new = set(_header(raw_dir, "23.05", "test1"))
    excluded = {"time", "timestamp", "attack", "attack_P1", "attack_P2", "attack_P3"}
    return [tag for tag in old if tag in new and tag not in excluded]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_stream(raw_dir: Path, version: str, name: str, tags: list[str]) -> Stream:
    path = raw_path(raw_dir, version, name)
    time_column = "time" if version == "21.03" else "timestamp"
    columns = [time_column, *tags]
    if version == "21.03":
        columns.append("attack")
    frame = pd.read_csv(path, usecols=columns)
    timestamps = pd.to_datetime(frame[time_column], format="%Y-%m-%d %H:%M:%S")
    if version == "21.03":
        labels = frame["attack"]
        label_alignment = "embedded attack column"
    else:
        label_path = raw_dir / f"hai-{version}" / f"label-{name}.csv"
        label_frame = pd.read_csv(label_path)
        label_time = pd.to_datetime(label_frame["timestamp"], format="mixed")
        observed_time = timestamps.reset_index(drop=True)
        if observed_time.equals(label_time):
            label_alignment = "exact second timestamps"
        elif observed_time.dt.floor("min").equals(label_time):
            label_alignment = "minute timestamps verified against floored telemetry"
        else:
            raise ValueError(f"Timestamp mismatch between {path} and {label_path}")
        labels = label_frame["label"]
    epoch_seconds = timestamps.to_numpy().astype("datetime64[s]").astype(np.int64)
    if len(epoch_seconds) > 1 and not np.all(np.diff(epoch_seconds) == 1):
        raise ValueError(f"{version}/{name} is not a continuous one-second stream")
    values = frame[tags].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{version}/{name} contains NaN or infinite telemetry")
    y = labels.to_numpy(dtype=np.uint8)
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{version}/{name} labels are not binary")
    if name.startswith("train") and y.any():
        raise ValueError(f"Training file {version}/{name} contains attacks")
    return Stream(version, name, values, y, epoch_seconds, label_alignment=label_alignment)


def prepare(raw_dir: Path, processed_dir: Path, validation_fraction: float) -> dict[str, Any]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    tags = common_tags(raw_dir)
    shared_tags = tags.copy()
    streams: dict[tuple[str, str], Stream] = {}
    for version, groups in FILES.items():
        for names in groups.values():
            for name in names:
                streams[(version, name)] = _read_stream(raw_dir, version, name, tags)

    sums = np.zeros(len(tags), dtype=np.float64)
    squares = np.zeros(len(tags), dtype=np.float64)
    count = 0
    for name in FILES["21.03"]["train"]:
        stream = streams[("21.03", name)]
        stream.train_end = int(np.floor(len(stream) * (1.0 - validation_fraction)))
        train = stream.x[: stream.train_end].astype(np.float64)
        sums += train.sum(axis=0)
        squares += np.square(train).sum(axis=0)
        count += len(train)
    mean = sums / count
    variance = np.maximum(squares / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    retained = scale >= 1e-6
    if not retained.any():
        raise ValueError("All shared tags are constant on the source training partitions")
    excluded = [tag for tag, keep in zip(tags, retained, strict=True) if not keep]
    if excluded:
        for stream in streams.values():
            stream.x = stream.x[:, retained]
        tags = [tag for tag, keep in zip(tags, retained, strict=True) if keep]
        mean = mean[retained]
        scale = scale[retained]

    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "shared_tags": shared_tags,
        "shared_tag_count": len(shared_tags),
        "excluded_constant_tags": excluded,
        "source_std_threshold": 1e-6,
        "common_tags": tags,
        "common_tag_count": len(tags),
        "validation_fraction": validation_fraction,
        "scaler": "source-only zero-variance filter; mean/std fitted on chronological 21.03 training partitions",
        "files": {},
    }
    for (version, name), stream in streams.items():
        normalized = (stream.x - mean.astype(np.float32)) / scale.astype(np.float32)
        output = processed_dir / f"{version}_{name}.npz"
        np.savez_compressed(
            output,
            x=normalized.astype(np.float32),
            y=stream.y,
            timestamps=stream.timestamps,
            train_end=-1 if stream.train_end is None else stream.train_end,
        )
        changes = np.diff(np.concatenate(([0], stream.y.astype(np.int8))))
        metadata["files"][f"{version}/{name}"] = {
            "rows": len(stream),
            "start": str(pd.to_datetime(stream.timestamps[0], unit="s")),
            "end": str(pd.to_datetime(stream.timestamps[-1], unit="s")),
            "attack_points": int(stream.y.sum()),
            "attack_events": int(np.sum(changes == 1)),
            "train_end": stream.train_end,
            "label_alignment": stream.label_alignment,
            "sha256": _sha256(raw_path(raw_dir, version, name)),
        }
    np.savez(processed_dir / "scaler.npz", mean=mean, scale=scale, tags=np.asarray(tags))
    (processed_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_stream(processed_dir: Path, version: str, name: str) -> Stream:
    with np.load(processed_dir / f"{version}_{name}.npz") as data:
        train_end = int(data["train_end"])
        return Stream(
            version=version,
            name=name,
            x=data["x"],
            y=data["y"],
            timestamps=data["timestamps"],
            train_end=None if train_end < 0 else train_end,
        )


def load_tags(processed_dir: Path) -> list[str]:
    return json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))["common_tags"]


def training_ends(stream: Stream, window: int, stride: int) -> Array:
    if stream.train_end is None:
        raise ValueError(f"{stream.version}/{stream.name} has no training partition")
    return np.arange(window - 1, stream.train_end, stride, dtype=np.int64)


def evaluation_ends(stream: Stream, window: int, stride: int, calibration: bool = False) -> Array:
    start = stream.train_end if calibration else 0
    if start is None:
        raise ValueError(f"{stream.version}/{stream.name} has no calibration partition")
    return np.arange(max(start, window - 1), len(stream), stride, dtype=np.int64)


def extract_windows(x: Array, ends: Array, window: int) -> Array:
    """Return (windows, channels, time); each score is aligned to its causal end point."""
    indices = ends[:, None] - window + 1 + np.arange(window, dtype=np.int64)[None, :]
    return x[indices].transpose(0, 2, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    metadata = prepare(
        Path(config["data"]["raw_dir"]),
        Path(config["data"]["processed_dir"]),
        float(config["data"]["validation_fraction"]),
    )
    files = metadata["files"]
    print(f"Prepared {len(files)} streams with {metadata['common_tag_count']} common tags")
    for name, info in files.items():
        print(name, info["rows"], "rows", info["attack_events"], "attack events")


if __name__ == "__main__":
    main()
