#!/usr/bin/env python3
"""
Train a generated SNN model on Google Speech Commands.

The script is based on the training flow from
`config/res_google_1_to_3_to_1.ipynb`, but packaged as a reusable CLI:

    python config/snn_coderun.py config/results/model_ladder.py

It will:
1. Load a model class from the provided `.py` file.
2. Load a cached mel dataset or recreate it from raw audio.
3. Train for 15 epochs.
4. Evaluate on the test split.
5. Save a checkpoint with the trained parameters and metrics.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import random
import re
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from speech_dataset import (
    N_MELS,
    create_dataset,
    default_cache_path,
    is_cached_dataset_path,
    load_dataset_cache,
    resolve_raw_data_path,
    save_dataset_cache,
)


TIMESTEPS = 20
DEFAULT_EPOCHS = 50
DEFAULT_LR = 2e-3
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 2
DEFAULT_RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an SNN model implementation on Google Speech Commands."
    )
    parser.add_argument(
        "model_py",
        type=Path,
        help="Path to a Python file with an nn.Module implementation.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=(
            "Path to a raw Google Speech Commands dataset directory or to a cached `.npz` "
            "file. If omitted, a local cache in `data/` is preferred, otherwise kagglehub is used."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TIMESTEPS,
        help=f"Number of rate-coding timesteps (default: {TIMESTEPS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning rate (default: {DEFAULT_LR}).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"DataLoader workers (default: {DEFAULT_NUM_WORKERS}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, for example `cpu`, `cuda`, or `cuda:0`.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed (default: {DEFAULT_RANDOM_SEED}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output checkpoint path. Defaults to `<model_stem>_trained.pt` in the repo root.",
    )
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default=None,
        help="Optional JSON or Python dict with model constructor overrides.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dataset_path(data_path: Path | None) -> Path:
    if data_path is not None:
        resolved = data_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {resolved}")
        if resolved.is_file() and not is_cached_dataset_path(resolved):
            raise ValueError(
                f"Expected a raw dataset directory or a `.npz` cache file, but got: {resolved}"
            )
        return resolved

    cached = default_cache_path()
    if cached.exists():
        return cached.resolve()
    return resolve_raw_data_path(None)


def load_dataset_bundle(
    data_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, Any], str]:
    if is_cached_dataset_path(data_path):
        X, y, label_map, metadata = load_dataset_cache(data_path)
        return X, y, label_map, metadata, "cache"

    X, y, label_map = create_dataset(data_path)
    cache_path = default_cache_path()
    saved_cache_path: Path | None = None
    if not cache_path.exists():
        saved_cache_path = save_dataset_cache(
            cache_path,
            X,
            y,
            label_map,
            source_path=data_path,
        )
    metadata = {
        "source_path": str(data_path),
        "num_samples": int(X.shape[0]),
        "feature_shape": list(X.shape[1:]),
        "num_classes": len(label_map),
    }
    if saved_cache_path is not None:
        metadata["cache_path"] = str(saved_cache_path)
    return X, y, label_map, metadata, "raw"


def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    raise ValueError("Expected 2D or 3D tensor (H x W or C x H x W).")


@dataclass
class TimeMask:
    enabled: bool = True
    num_masks: int = 2
    max_width_frac: float = 0.12
    fill: str = "min"

    def __call__(self, x: torch.Tensor, other: Any = None) -> torch.Tensor:
        if not self.enabled:
            return x
        x = _ensure_3d(x)
        _, _, width = x.shape
        fill_value = x.min() if self.fill == "min" else torch.tensor(0.0, device=x.device, dtype=x.dtype)
        max_w = max(1, int(width * self.max_width_frac))
        for _ in range(self.num_masks):
            w = random.randint(1, max_w)
            t0 = random.randint(0, max(0, width - w))
            x[:, :, t0 : t0 + w] = fill_value
        return x


@dataclass
class FreqMask:
    enabled: bool = True
    num_masks: int = 2
    max_width_frac: float = 0.12
    fill: str = "min"

    def __call__(self, x: torch.Tensor, other: Any = None) -> torch.Tensor:
        if not self.enabled:
            return x
        x = _ensure_3d(x)
        _, height, _ = x.shape
        fill_value = x.min() if self.fill == "min" else torch.tensor(0.0, device=x.device, dtype=x.dtype)
        max_h = max(1, int(height * self.max_width_frac))
        for _ in range(self.num_masks):
            h = random.randint(1, max_h)
            f0 = random.randint(0, max(0, height - h))
            x[:, f0 : f0 + h, :] = fill_value
        return x


@dataclass
class GaussianNoise:
    enabled: bool = True
    std: float = 0.02

    def __call__(self, x: torch.Tensor, other: Any = None) -> torch.Tensor:
        if not self.enabled or self.std <= 0:
            return x
        x = _ensure_3d(x)
        return x + torch.randn_like(x) * self.std


@dataclass
class SaltPepperPixelwise:
    enabled: bool = True
    p: float = 0.01
    salt_vs_pepper: float = 0.5
    salt_value: str = "max"
    pepper_value: str = "min"
    salt_fixed: float = 1.0
    pepper_fixed: float = 0.0

    def __call__(self, x: torch.Tensor, other: Any = None) -> torch.Tensor:
        if not self.enabled or self.p <= 0:
            return x

        x = _ensure_3d(x)
        channels, height, width = x.shape
        noise_mask = torch.rand((height, width), device=x.device) < self.p
        salt_mask = noise_mask & (torch.rand((height, width), device=x.device) < self.salt_vs_pepper)
        pepper_mask = noise_mask & (~salt_mask)

        salt_val = (
            x.max()
            if self.salt_value == "max"
            else torch.tensor(self.salt_fixed, device=x.device, dtype=x.dtype)
        )
        pepper_val = (
            x.min()
            if self.pepper_value == "min"
            else torch.tensor(self.pepper_fixed, device=x.device, dtype=x.dtype)
        )

        for c in range(channels):
            xc = x[c]
            xc[salt_mask] = salt_val
            xc[pepper_mask] = pepper_val
            x[c] = xc
        return x


class SpectrogramAugment:
    def __init__(
        self,
        time_mask: TimeMask = TimeMask(),
        freq_mask: FreqMask = FreqMask(),
        gauss: GaussianNoise = GaussianNoise(),
        sp: SaltPepperPixelwise = SaltPepperPixelwise(),
        prob: float = 0.95,
    ) -> None:
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.gauss = gauss
        self.sp = sp
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.prob < 1.0 and random.random() > self.prob:
            return _ensure_3d(x)
        x = _ensure_3d(x)
        x = self.freq_mask(x)
        x = self.time_mask(x)
        x = self.gauss(x)
        x = self.sp(x)
        return x


def apply_offline_augmentation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    transform: SpectrogramAugment,
    ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    if ratio <= 0:
        return X_train, y_train

    offline_transform = SpectrogramAugment(
        prob=1.0,
        time_mask=transform.time_mask,
        freq_mask=transform.freq_mask,
        gauss=transform.gauss,
        sp=transform.sp,
    )

    n = len(X_train)
    m = int(n * ratio)
    if m <= 0:
        return X_train, y_train

    indices = np.random.choice(n, size=m, replace=False)
    X_aug: list[np.ndarray] = []
    y_aug: list[int] = []
    for idx in tqdm(indices, desc="Offline augmentation", leave=False):
        x = torch.tensor(X_train[idx], dtype=torch.float32)
        x_aug = offline_transform(x)
        X_aug.append(x_aug.squeeze(0).cpu().numpy())
        y_aug.append(int(y_train[idx]))

    X_extra = np.stack(X_aug, axis=0)
    y_extra = np.asarray(y_aug, dtype=np.int64)
    return (
        np.concatenate([X_train, X_extra], axis=0),
        np.concatenate([y_train, y_extra], axis=0),
    )


class SimpleSpectrogramDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, transform: Any = None) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self.X[idx]
        if self.transform is not None:
            out = self.transform(spec)
            if out.dim() == 2:
                spec = out.unsqueeze(0)
            elif out.dim() == 3:
                spec = out
            else:
                raise ValueError("Unexpected transform output dims.")
        else:
            spec = spec.unsqueeze(0)
        return spec, self.y[idx]

    def __len__(self) -> int:
        return len(self.y)


def build_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        # stratify=y,
    )

    train_transform = SpectrogramAugment(
        prob=0.01,
        time_mask=TimeMask(enabled=False, num_masks=2, max_width_frac=0.12, fill="min"),
        freq_mask=FreqMask(enabled=False, num_masks=2, max_width_frac=0.12, fill="min"),
        gauss=GaussianNoise(enabled=False, std=0.02),
        sp=SaltPepperPixelwise(enabled=True, p=0.01, salt_vs_pepper=0.5),
    )
    test_transform = None

    X_train, y_train = apply_offline_augmentation(X_train, y_train, train_transform, ratio=0.5)

    train_dataset = SimpleSpectrogramDataset(X_train, y_train, transform=train_transform)
    test_dataset = SimpleSpectrogramDataset(X_test, y_test, transform=test_transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    split_info = {
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
    }
    return train_loader, test_loader, split_info


def rate_code(
    x: torch.Tensor,
    time_steps: int,
    eps: float = 1e-6,
    p_max: float = 0.9,
    gamma: float = 1.5,
) -> torch.Tensor:
    if x.dim() > 2:
        x = x.view(x.size(0), -1)

    x_min = x.amin(dim=1, keepdim=True)
    x_max = x.amax(dim=1, keepdim=True)
    denom = (x_max - x_min).clamp_min(eps)
    p = (x - x_min) / denom

    if gamma != 1.0:
        p = p.clamp(0.0, 1.0).pow(gamma)
    else:
        p = p.clamp(0.0, 1.0)

    if p_max is not None:
        p = p.clamp_max(p_max)

    probs = p.detach().unsqueeze(0).expand(time_steps, -1, -1)
    spikes = (
        torch.rand(time_steps, *p.shape, device=x.device, dtype=p.dtype) < probs
    ).to(torch.float32)
    return spikes


def patch_common_model_issues(source: str) -> str:
    # Current generated CNN models use `self._cnn_out_size` in a few places,
    # while the actual attribute that gets assigned is `self.cnn_out_size`.
    return source.replace("self._cnn_out_size", "self.cnn_out_size")


def load_model_module(model_path: Path) -> tuple[types.ModuleType, str]:
    source = model_path.read_text(encoding="utf-8")
    source = patch_common_model_issues(source)

    module = types.ModuleType(f"snn_dynamic_{model_path.stem}")
    module.__file__ = str(model_path)
    exec(compile(source, str(model_path), "exec"), module.__dict__)
    return module, source


def extract_model_name(source: str) -> str | None:
    match = re.search(r"^\s*#\s*Model:\s*([A-Za-z_][A-Za-z0-9_]*)\s*$", source, re.MULTILINE)
    return match.group(1) if match else None


def extract_example_kwargs(source: str, class_name: str) -> dict[str, Any] | None:
    pattern = re.compile(
        rf"^\s*#\s*model\s*=\s*{re.escape(class_name)}\((.+)\)\s*$",
        re.MULTILINE,
    )
    match = pattern.search(source)
    if not match:
        return None

    call = ast.parse(f"{class_name}({match.group(1)})", mode="eval").body
    if not isinstance(call, ast.Call):
        return None

    kwargs: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue
        kwargs[kw.arg] = ast.literal_eval(kw.value)
    return kwargs


def pick_model_class(module: types.ModuleType, source: str) -> type[nn.Module]:
    class_name = extract_model_name(source)
    if class_name is not None:
        candidate = getattr(module, class_name, None)
        if inspect.isclass(candidate) and issubclass(candidate, nn.Module):
            return candidate

    module_classes = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, nn.Module)
        and obj.__module__ == module.__name__
        and obj.__name__ not in {"BimodalLIFLayer", "SpikingConvBlock", "SpikingConvFrontEnd"}
    ]
    if not module_classes:
        raise RuntimeError("No nn.Module subclass found in the provided model file.")
    return module_classes[-1]


def infer_model_kwargs(model_cls: type[nn.Module]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    signature = inspect.signature(model_cls.__init__)
    for name, param in list(signature.parameters.items())[1:]:
        if param.default is not inspect.Parameter.empty:
            continue
        if name in {"inp_size", "input_size"}:
            kwargs[name] = N_MELS * 32
        elif name in {"out_size", "output_size"}:
            kwargs[name] = 30
        elif name.endswith("_size") or name.endswith("size"):
            kwargs[name] = 256
        else:
            raise RuntimeError(
                f"Could not infer required constructor argument `{name}` for {model_cls.__name__}. "
                "Pass explicit overrides with --model-kwargs."
            )
    return kwargs


def load_model(
    model_path: Path,
    device: torch.device,
    model_kwargs_override: dict[str, Any] | None,
) -> tuple[nn.Module, str, dict[str, Any]]:
    module, source = load_model_module(model_path)
    model_cls = pick_model_class(module, source)
    model_kwargs = extract_example_kwargs(source, model_cls.__name__) or infer_model_kwargs(model_cls)
    if model_kwargs_override:
        model_kwargs.update(model_kwargs_override)

    model = model_cls(**model_kwargs).to(device)
    return model, model_cls.__name__, model_kwargs


def smoke_test_model(
    model: nn.Module,
    model_kwargs: dict[str, Any],
    device: torch.device,
    timesteps: int,
) -> None:
    input_features = model_kwargs.get("inp_size", model_kwargs.get("input_size", N_MELS * 32))
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(timesteps, 2, input_features, device=device)
        outputs = model(dummy)
    if outputs.dim() == 2:
        outputs = outputs.unsqueeze(0)
    if outputs.dim() != 3:
        raise RuntimeError(
            f"Smoke test failed: expected [T, B, C] or [B, C], got {tuple(outputs.shape)}."
        )


def format_firing_rates(firing_rates: dict[str, float]) -> str:
    if not firing_rates:
        return ""
    parts = [f"{key}={value:.3f}" for key, value in firing_rates.items()]
    return " | FR " + " ".join(parts)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    timesteps: int,
    train: bool,
) -> tuple[float, float, dict[str, float]]:
    total_loss = 0.0
    correct = 0
    count = 0
    firing_acc: dict[str, float] = {}
    n_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        images_flat = images.squeeze(1).flatten(1)
        spikes = rate_code(images_flat, timesteps)

        if train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(spikes)
        if outputs.dim() == 2:
            outputs = outputs.unsqueeze(0)
        if outputs.dim() != 3:
            raise RuntimeError(
                f"Model output must have shape [T, B, C] or [B, C], got {tuple(outputs.shape)}."
            )

        firing_rates = getattr(model, "last_firing_rates", None)
        if isinstance(firing_rates, dict):
            if not firing_acc:
                firing_acc = {key: 0.0 for key in firing_rates}
            for key, value in firing_rates.items():
                firing_acc[key] += float(value)
            n_batches += 1

        log_probs = F.log_softmax(outputs, dim=-1)
        summed = torch.logsumexp(log_probs, dim=0)
        loss = F.nll_loss(summed, labels)

        if train:
            loss.backward()
            for name, param in model.named_parameters():
                # if param.grad is not None and "tau" not in name.lower():
                if param.grad is not None:
                    param.data.clamp_(-10.0, 10.0)
            optimizer.step()

        total_loss += float(loss.item())
        correct += int((summed.argmax(dim=1) == labels).sum().item())
        count += int(labels.size(0))

    avg_firing = (
        {key: value / max(n_batches, 1) for key, value in firing_acc.items()}
        if firing_acc
        else {}
    )
    return total_loss / max(len(loader), 1), correct / max(count, 1), avg_firing


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    timesteps: int,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_acc, train_fr = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            timesteps=timesteps,
            train=True,
        )

        model.eval()
        with torch.no_grad():
            test_loss, test_acc, test_fr = run_epoch(
                model=model,
                loader=test_loader,
                optimizer=optimizer,
                device=device,
                timesteps=timesteps,
                train=False,
            )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch:>2}/{epochs} | "
            f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"Test loss {test_loss:.4f} acc {test_acc:.4f}"
            f"{format_firing_rates(test_fr or train_fr)}"
        )

    return {
        "history": history,
        "final_test_accuracy": history["test_acc"][-1],
        "best_test_accuracy": max(history["test_acc"]),
    }


def build_checkpoint_path(model_path: Path, output: Path | None) -> Path:
    if output is not None:
        return output.expanduser().resolve()
    repo_root = Path(__file__).resolve().parent
    return repo_root / f"{model_path.stem}_trained.pt"


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    model_path = args.model_py.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    device = choose_device(args.device)
    model_kwargs_override = json_loads_if_needed(args.model_kwargs)

    print(f"Loading model from: {model_path}")
    model, model_name, model_kwargs = load_model(
        model_path=model_path,
        device=device,
        model_kwargs_override=model_kwargs_override,
    )
    smoke_test_model(model, model_kwargs, device, args.timesteps)
    print(f"Loaded model: {model_name}")
    print(f"Model kwargs: {model_kwargs}")

    data_path = resolve_dataset_path(args.data_path)
    print(f"Dataset path: {data_path}")
    X, y, label_map, dataset_metadata, dataset_kind = load_dataset_bundle(data_path)
    if dataset_kind == "cache":
        print(f"Cached dataset loaded: X={X.shape}, y={y.shape}, classes={len(label_map)}")
    else:
        print(f"Dataset built: X={X.shape}, y={y.shape}, classes={len(label_map)}")
        cache_path = dataset_metadata.get("cache_path")
        if cache_path:
            print(f"Cached dataset saved to: {cache_path}")

    train_loader, test_loader, split_info = build_dataloaders(
        X=X,
        y=y,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(
        f"Train/Test split prepared: "
        f"{split_info['train_size']} train / {split_info['test_size']} test"
    )

    results = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        timesteps=args.timesteps,
    )

    checkpoint_path = build_checkpoint_path(model_path, args.output)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": model_name,
        "model_path": str(model_path),
        "model_kwargs": model_kwargs,
        "state_dict": model.state_dict(),
        "test_accuracy": results["final_test_accuracy"],
        "best_test_accuracy": results["best_test_accuracy"],
        "history": results["history"],
        "label_map": label_map,
        "dataset_path": str(data_path),
        "dataset_kind": dataset_kind,
        "dataset_metadata": dataset_metadata,
        "timesteps": args.timesteps,
        "epochs": args.epochs,
        "learning_rate": args.lr,
    }
    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Final test accuracy: {results['final_test_accuracy']:.4f}")
    print(f"Best test accuracy:  {results['best_test_accuracy']:.4f}")
    return 0


def json_loads_if_needed(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--model-kwargs must be a dict.")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
