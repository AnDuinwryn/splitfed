from __future__ import annotations

import numpy as np
import librosa


def normalize_mel(x: np.ndarray, delta: int = 0, norm_mode: str = "per sample") -> np.ndarray:
    """Apply log10 scaling, optional deltas, and z-normalization."""
    mel_log = np.log10(x + 1e-10)

    if delta == 2:
        delta_1 = librosa.feature.delta(mel_log, order=1, mode="nearest")
        delta_2 = librosa.feature.delta(mel_log, order=2, mode="nearest")
        x_log = np.concatenate((mel_log, delta_1, delta_2), axis=3)
    elif delta == 1:
        delta_1 = librosa.feature.delta(mel_log, order=1, mode="nearest")
        x_log = np.concatenate((mel_log, delta_1), axis=3)
    else:
        x_log = mel_log.copy()

    x_norm = x_log.copy()

    if norm_mode == "per sample":
        for aud in range(x_log.shape[0]):
            for f in range(x_log.shape[-1]):
                for freq in range(x_log.shape[1]):
                    mean = np.mean(x_log[aud, freq, :, f])
                    std = np.std(x_log[aud, freq, :, f])
                    denom = std if std > 0 else 1.0
                    x_norm[aud, freq, :, f] = (x_log[aud, freq, :, f] - mean) / denom
        return x_norm

    if norm_mode == "training data":
        spec_mean = np.zeros((128, x_log.shape[-1]))
        spec_std = np.zeros((128, x_log.shape[-1]))
        for freq in range(x_log.shape[1]):
            for f in range(x_log.shape[-1]):
                spec_mean[freq, f] = np.mean(x_log[:, freq, :, f])
                spec_std[freq, f] = np.std(x_log[:, freq, :, f])
                denom = spec_std[freq, f] if spec_std[freq, f] > 0 else 1.0
                x_norm[:, freq, :, f] = (x_log[:, freq, :, f] - spec_mean[freq, f]) / denom
        return x_norm, spec_mean, spec_std

    return x_norm


def mel_first_channel(x: np.ndarray) -> np.ndarray:
    """Take first mel channel: (N, F, T, C) -> (N, F, T, 1)."""
    return x[:, :, :, [0]]
