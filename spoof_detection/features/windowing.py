"""
windowing.py

Corta una señal de audio en ventanas sucesivas de duración configurable,
con o sin solapamiento. Es la pieza base para simular procesamiento en
tiempo real: en vez de analizar el audio completo, lo vamos "troceando"
como llegaría en un flujo continuo (streaming).

Uso típico:
    from features.windowing import make_windows

    windows = make_windows(audio, sr=16000, window_sec=1.0, hop_sec=0.5)
    # windows es una lista de arreglos numpy, cada uno de duración ~1.0s
"""

import numpy as np


def make_windows(audio: np.ndarray, sr: int, window_sec: float,
                  hop_sec: float | None = None, drop_last: bool = True):
    """
    Parameters
    ----------
    audio : np.ndarray
        Señal de audio mono, 1D.
    sr : int
        Frecuencia de muestreo en Hz.
    window_sec : float
        Duración de cada ventana en segundos (ej. 0.5, 1, 2, 3, 5).
    hop_sec : float, opcional
        Cada cuánto se desliza la ventana. Si es None, se usa hop_sec =
        window_sec (sin solapamiento). Un hop_sec menor que window_sec
        produce solapamiento (overlap).
    drop_last : bool
        Si True, descarta la última ventana si queda incompleta.
        Si False, la rellena con ceros (zero-padding) hasta completarla.

    Returns
    -------
    list[np.ndarray]
        Lista de ventanas, cada una de largo int(window_sec * sr).
    """
    if hop_sec is None:
        hop_sec = window_sec

    window_len = int(round(window_sec * sr))
    hop_len = int(round(hop_sec * sr))

    if window_len <= 0 or hop_len <= 0:
        raise ValueError("window_sec y hop_sec deben ser positivos")

    windows = []
    start = 0
    n = len(audio)
    while start < n:
        end = start + window_len
        if end <= n:
            windows.append(audio[start:end])
        else:
            if drop_last:
                break
            chunk = audio[start:n]
            pad = np.zeros(window_len - len(chunk), dtype=audio.dtype)
            windows.append(np.concatenate([chunk, pad]))
            break
        start += hop_len

    return windows


def window_count(duration_sec: float, window_sec: float, hop_sec: float | None = None) -> int:
    """Calcula cuántas ventanas completas caben en un audio de cierta duración,
    sin tener que generarlas. Útil para reportar estadísticas rápido."""
    if hop_sec is None:
        hop_sec = window_sec
    if duration_sec < window_sec:
        return 0
    return int(np.floor((duration_sec - window_sec) / hop_sec)) + 1
