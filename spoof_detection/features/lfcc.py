"""
lfcc.py

Extracción de LFCC (Linear Frequency Cepstral Coefficients): el Enfoque A
(interpretable) del proyecto.

Pipeline: señal --> STFT --> banco de filtros lineal --> log --> DCT --> LFCC

A diferencia de los MFCC (que usan un banco de filtros mel, más denso en
graves), aquí el banco de filtros tiene bandas de ancho IGUAL en Hz.

Implementación con numpy/scipy únicamente.
"""

import numpy as np
from scipy.signal import stft
from scipy.fft import dct


def _linear_filterbank(n_filters: int, n_fft: int, sr: int,
                        fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """Construye un banco de filtros triangulares con bordes igualmente
    espaciados en Hz (a diferencia del banco mel, que los espacia en una
    escala perceptual no lineal)."""
    fmax = fmax or sr / 2
    n_bins = n_fft // 2 + 1

    freq_points = np.linspace(fmin, fmax, n_filters + 2)
    bin_points = np.floor((n_fft + 1) * freq_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, n_bins - 1)

    fb = np.zeros((n_filters, n_bins))
    for i in range(1, n_filters + 1):
        left, center, right = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        if center > left:
            fb[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb


def extract_lfcc(audio: np.ndarray, sr: int, n_lfcc: int = 20,
                  n_filters: int = 40, n_fft: int = 512,
                  win_length_sec: float = 0.025, hop_length_sec: float = 0.010) -> np.ndarray:
    """
    Extrae LFCC de una señal.

    Returns
    -------
    np.ndarray de forma (n_frames, n_lfcc)
        Un vector de n_lfcc coeficientes por cada trama corta (~25 ms)
        dentro de la ventana.
    """
    win_length = int(round(win_length_sec * sr))
    hop_length = int(round(hop_length_sec * sr))
    noverlap = max(win_length - hop_length, 0)

    if n_fft < win_length:
        n_fft = 1 << (win_length - 1).bit_length()

    if len(audio) < win_length:
        audio = np.pad(audio, (0, win_length - len(audio)))

    _, _, Zxx = stft(audio, fs=sr, nperseg=win_length, noverlap=noverlap, nfft=n_fft)
    power_spec = np.abs(Zxx) ** 2  # (n_bins, n_frames)

    fb = _linear_filterbank(n_filters, n_fft, sr)
    fb_energy = fb @ power_spec  # (n_filters, n_frames)
    log_energy = np.log(fb_energy + 1e-10)

    lfcc = dct(log_energy, type=2, axis=0, norm="ortho")[:n_lfcc]  # (n_lfcc, n_frames)
    return lfcc.T  # (n_frames, n_lfcc)


def lfcc_window_features(audio: np.ndarray, sr: int, **lfcc_kwargs) -> np.ndarray:
    """
    Reduce los LFCC por trama de una ventana a un solo vector fijo,
    concatenando media y desviación estándar a través del tiempo, 
    lo que entra al SVM (necesita vectores de tamaño
    fijo, sin importar la duración de la ventana).
    """
    lfcc = extract_lfcc(audio, sr, **lfcc_kwargs)  # (n_frames, n_lfcc)
    mean = lfcc.mean(axis=0)
    std = lfcc.std(axis=0)
    return np.concatenate([mean, std])
