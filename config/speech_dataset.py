#!/usr/bin/env python3
"""
Utilities for building and caching the Google Speech Commands mel dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from tqdm import tqdm


SAMPLE_RATE = 16_000
DURATION = 1.0
N_MELS = 128
N_FFT = 400
GSC_DATASET_ID = "neehakurelli/google-speech-commands"
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac"}
DEFAULT_CACHE_BASENAME = "google_speech_commands_mels.npz"


def default_cache_path(project_root: Path | None = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    return project_root / "data" / DEFAULT_CACHE_BASENAME


def is_cached_dataset_path(path: Path) -> bool:
    return path.suffix.lower() == ".npz"


def resolve_raw_data_path(data_path: Path | None) -> Path:
    if data_path is not None:
        resolved = data_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {resolved}")
        if resolved.is_file():
            raise ValueError(
                f"Expected a source dataset directory, but got a file path: {resolved}"
            )
        return resolved

    import kagglehub

    downloaded = Path(kagglehub.dataset_download(GSC_DATASET_ID)).resolve()
    if not downloaded.exists():
        raise FileNotFoundError(f"kagglehub returned a missing dataset path: {downloaded}")
    return downloaded


def load_audio(
    file_path: Path,
    target_sr: int = SAMPLE_RATE,
    duration: float = DURATION,
) -> np.ndarray:
    audio, _ = librosa.load(str(file_path), sr=target_sr, duration=duration)
    audio = librosa.util.fix_length(audio, size=int(target_sr * duration))
    return audio


def extract_features(file_path: Path) -> np.ndarray:
    audio = load_audio(file_path)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db.astype(np.float32)


def create_dataset(
    data_path: Path,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    X: list[np.ndarray] = []
    y: list[int] = []

    classes = sorted(
        item.name
        for item in data_path.iterdir()
        if item.is_dir() and item.name != "_background_noise_"
    )
    if not classes:
        raise RuntimeError(f"No class folders found in dataset path: {data_path}")

    label_map = {label: idx for idx, label in enumerate(classes)}
    class_iter = tqdm(classes, desc="Classes", disable=not show_progress)

    for label in class_iter:
        folder = data_path / label
        audio_files = sorted(
            path for path in folder.iterdir() if path.suffix.lower() in AUDIO_SUFFIXES
        )
        file_iter = tqdm(
            audio_files,
            desc=f"Features:{label}",
            leave=False,
            disable=not show_progress,
        )
        for file_path in file_iter:
            X.append(extract_features(file_path))
            y.append(label_map[label])

    if not X:
        raise RuntimeError(f"No audio files found under dataset path: {data_path}")

    return np.stack(X, axis=0), np.asarray(y, dtype=np.int64), label_map


def save_dataset_cache(
    output_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    label_map: dict[str, int],
    *,
    source_path: Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_names = np.asarray(
        [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]
    )
    payload = {
        "dataset_id": GSC_DATASET_ID,
        "source_path": str(source_path),
        "sample_rate": SAMPLE_RATE,
        "duration": DURATION,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "num_samples": int(X.shape[0]),
        "feature_shape": list(X.shape[1:]),
        "num_classes": len(label_map),
    }
    if metadata:
        payload.update(metadata)

    np.savez_compressed(
        output_path,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        label_names=label_names,
        metadata_json=np.asarray(json.dumps(payload), dtype=np.str_),
    )
    return output_path


def load_dataset_cache(
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, Any]]:
    cache_path = cache_path.expanduser().resolve()
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached dataset file does not exist: {cache_path}")

    with np.load(cache_path, allow_pickle=False) as bundle:
        X = np.asarray(bundle["X"], dtype=np.float32)
        y = np.asarray(bundle["y"], dtype=np.int64)
        if "label_names" not in bundle:
            raise KeyError(f"Cached dataset is missing `label_names`: {cache_path}")
        label_names = [str(label) for label in bundle["label_names"].tolist()]
        metadata_json = ""
        if "metadata_json" in bundle:
            metadata_json = str(bundle["metadata_json"].item())

    label_map = {label: idx for idx, label in enumerate(label_names)}
    metadata = json.loads(metadata_json) if metadata_json else {}
    return X, y, label_map, metadata
