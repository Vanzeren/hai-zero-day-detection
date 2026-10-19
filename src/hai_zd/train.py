"""Train per-sensor anomaly detectors on the normal 21.03 partition and score all streams.

Only the chronological training partitions of the 21.03 train files are used for
fitting; attack labels are never touched during training or model selection.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

Array = NDArray[Any]
Config = dict[str, Any]

from hai_zd.data import FILES, evaluation_ends, extract_windows, load_stream, training_ends
from hai_zd.models import (
    ConvVAE,
    DeepSVDD,
    PCAReconstructionBaseline,
    svdd_sensor_scores,
    vae_sensor_scores,
)

SMOKE_EPOCHS = 1
SMOKE_STEPS_PER_EPOCH = 2
SMOKE_SCORE_WINDOWS = 8


def load_config(path: Path) -> Config:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def training_windows(config: Config) -> Array:
    """All non-overlapping normal windows from the 21.03 train partitions: (N, C, T)."""
    processed_dir = Path(config["data"]["processed_dir"])
    window, stride = config["window"]["length"], config["window"]["train_stride"]
    parts = []
    for name in FILES["21.03"]["train"]:
        stream = load_stream(processed_dir, "21.03", name)
        ends = training_ends(stream, window, stride)
        parts.append(extract_windows(stream.x, ends, window))
    return np.concatenate(parts, axis=0)


def _sensor_batches(
    windows: Array, steps: int, batch_size: int, rng: np.random.Generator
) -> list[torch.Tensor]:
    """Uniformly sample steps batches of (batch_size, 1, T) single-sensor sequences."""
    n_windows, n_channels, time_len = windows.shape
    batches = []
    for _ in range(steps):
        window_idx = rng.integers(0, n_windows, size=batch_size)
        channel_idx = rng.integers(0, n_channels, size=batch_size)
        sample = windows[window_idx, channel_idx, :].reshape(batch_size, 1, time_len)
        batches.append(torch.from_numpy(np.ascontiguousarray(sample)))
    return batches


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_svdd(
    windows: Array, config: Config, seed: int, epochs: int, steps_per_epoch: int
) -> tuple[DeepSVDD, Config]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_cfg = config["training"]
    model = DeepSVDD(width=config["model"]["width"], latent_dim=config["model"]["latent_dim"])

    # Official DeepSVDD center initialization: mean latent over the training data,
    # near-zero coordinates pushed to +/-0.1 (happens inside initialize_center).
    center_start = time.perf_counter()
    model.initialize_center(torch.from_numpy(windows.reshape(-1, 1, windows.shape[-1])))
    center_seconds = time.perf_counter() - center_start

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    history = []
    start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        losses = []
        for batch in _sensor_batches(windows, steps_per_epoch, train_cfg["batch_size"], rng):
            optimizer.zero_grad()
            loss = torch.mean(torch.sum((model(batch) - model.center) ** 2, dim=1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
    elapsed = time.perf_counter() - start
    model.eval()
    return model, {
        "losses": history,
        "train_seconds": elapsed,
        "center_init_seconds": center_seconds,
        "center_init": "mean of initial-network latents over all training windows; "
        "coordinates with |mean| < 0.1 set to +/-0.1 (DeepSVDD official)",
        "parameters": _parameter_count(model),
    }


def _elbo(x: torch.Tensor, reconstruction: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    recon = 0.5 * torch.sum((x - reconstruction) ** 2, dim=(1, 2))
    kl = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=1)
    return torch.mean(recon + kl)


def train_vae(
    windows: Array, config: Config, seed: int, epochs: int, steps_per_epoch: int
) -> tuple[ConvVAE, Config]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_cfg = config["training"]
    model = ConvVAE(
        width=config["model"]["width"],
        latent_dim=config["model"]["latent_dim"],
        window=config["window"]["length"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    history = []
    start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        losses = []
        for batch in _sensor_batches(windows, steps_per_epoch, train_cfg["batch_size"], rng):
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(batch)
            loss = _elbo(batch, reconstruction, mu, logvar)  # beta = 1
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
    elapsed = time.perf_counter() - start
    model.eval()
    return model, {"losses": history, "train_seconds": elapsed, "parameters": _parameter_count(model)}


def score_windows_neural(score_fn, model, windows: Array, batch_windows: int) -> Array:
    """Per-sensor scores for (N, C, T) windows, batched along N: returns (N, C)."""
    scores = []
    for start in range(0, len(windows), batch_windows):
        batch = torch.from_numpy(windows[start : start + batch_windows])
        scores.append(score_fn(model, batch).numpy())
    return np.concatenate(scores, axis=0)


def _assert_valid(scores: Array, context: str) -> None:
    if not (np.all(np.isfinite(scores)) and np.all(scores >= 0)):
        raise ValueError(f"invalid (non-finite or negative) scores for {context}")


def score_all_streams(
    model_name: str,
    model,
    config: Config,
    out_dir: Path,
    max_windows: int | None = None,
) -> None:
    """Score calibration ends for 21.03 train files and all evaluation ends for test files."""
    processed_dir = Path(config["data"]["processed_dir"])
    window = config["window"]["length"]
    stride = config["window"]["eval_stride"]
    batch_windows = config["training"]["score_batch_windows"]
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [("21.03", name, True) for name in FILES["21.03"]["train"]]
    for version in FILES:
        targets += [(version, name, False) for name in FILES[version]["test"]]

    for version, name, calibration in targets:
        stream = load_stream(processed_dir, version, name)
        ends = evaluation_ends(stream, window, stride, calibration=calibration)
        if max_windows is not None:
            ends = ends[:max_windows]
        windows = extract_windows(stream.x, ends, window)
        if model_name == "pca":
            sensor_scores = model.sensor_scores(windows)
        else:
            score_fn = svdd_sensor_scores if model_name == "svdd" else vae_sensor_scores
            sensor_scores = score_windows_neural(score_fn, model, windows, batch_windows)
        sensor_scores = sensor_scores.astype(np.float32)
        _assert_valid(sensor_scores, f"{model_name} {version}/{name}")
        np.savez(
            out_dir / f"{version}_{name}.npz",
            sensor_scores=sensor_scores,
            scores=sensor_scores.max(axis=1).astype(np.float32),
            ends=ends,
            timestamps=stream.timestamps[ends],
            y=stream.y[ends],
        )


def _run_neural(
    model_name: str, windows: Array, config: Config, seed: int, config_path: Path, smoke: bool
) -> None:
    root = Path("results/smoke" if smoke else "results")
    epochs = SMOKE_EPOCHS if smoke else config["training"]["epochs"]
    steps = SMOKE_STEPS_PER_EPOCH if smoke else config["training"]["steps_per_epoch"]
    trainer = train_svdd if model_name == "svdd" else train_vae
    model, info = trainer(windows, config, seed, epochs, steps)

    checkpoint_dir = root / "checkpoints" / f"{model_name}_seed{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    info.update(
        {
            "model": model_name,
            "seed": seed,
            "epochs": epochs,
            "steps_per_epoch": steps,
            "smoke": smoke,
            "config": config,
            "config_path": str(config_path),
        }
    )
    (checkpoint_dir / "training.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    score_dir = root / "scores" / f"{model_name}_seed{seed}"
    score_all_streams(model_name, model, config, score_dir, SMOKE_SCORE_WINDOWS if smoke else None)


def _run_pca(windows: Array, config: Config, seed: int, config_path: Path, smoke: bool) -> None:
    root = Path("results/smoke" if smoke else "results")
    start = time.perf_counter()
    model = PCAReconstructionBaseline(config["model"]["pca_components"], random_state=seed)
    model.fit(windows)
    elapsed = time.perf_counter() - start

    checkpoint_dir = root / "checkpoints" / f"pca_seed{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model.pca, handle)
    info = {
        "model": "pca",
        "seed": seed,
        "train_seconds": elapsed,
        "smoke": smoke,
        "config": config,
        "config_path": str(config_path),
    }
    (checkpoint_dir / "training.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    score_dir = root / "scores" / f"pca_seed{seed}"
    score_all_streams("pca", model, config, score_dir, SMOKE_SCORE_WINDOWS if smoke else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--model", choices=["svdd", "vae", "pca", "all"], required=True)
    parser.add_argument("--seed", type=int, default=None, help="override; default: all config seeds")
    parser.add_argument("--smoke", action="store_true", help="1 epoch / 2 steps, few scored windows, results/smoke")
    args = parser.parse_args()

    config = load_config(args.config)
    torch.set_num_threads(config["training"]["num_threads"])
    seeds = [args.seed] if args.seed is not None else list(config["training"]["seeds"])
    names = ["svdd", "vae", "pca"] if args.model == "all" else [args.model]

    windows = training_windows(config)
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        for name in names:
            print(f"training {name} seed={seed}{' (smoke)' if args.smoke else ''}", flush=True)
            if name == "pca":
                _run_pca(windows, config, seed, args.config, args.smoke)
            else:
                _run_neural(name, windows, config, seed, args.config, args.smoke)


if __name__ == "__main__":
    main()
