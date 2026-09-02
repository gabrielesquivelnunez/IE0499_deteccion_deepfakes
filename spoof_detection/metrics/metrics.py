"""
metrics.py

Métricas de clasificación estándar en anti-spoofing.

EER (Equal Error Rate) es la métrica más importante del área: es el punto
en el que la tasa de falsos positivos (marcar como sintético un audio
real, "false alarm") se iguala con la tasa de falsos negativos (marcar
como real un audio sintético, "miss"). Se prefiere sobre accuracy porque
no depende de fijar un umbral de decisión arbitrario de antemano.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def compute_eer(bona_fide_scores: np.ndarray, spoof_scores: np.ndarray) -> tuple[float, float]:
    """
    Calcula el Equal Error Rate.

    Parameters
    ----------
    bona_fide_scores : np.ndarray
        Puntajes del detector (score = probabilidad de ser SINTÉTICO) para
        audios reales (bona fide). Idealmente deberían ser bajos.
    spoof_scores : np.ndarray
        Puntajes para audios sintéticos. Idealmente deberían ser altos.

    Returns
    -------
    (eer, threshold) : tuple[float, float]
        eer en [0, 1] y el umbral de puntaje donde ocurre.
    """
    thresholds = np.unique(np.concatenate([bona_fide_scores, spoof_scores]))
    thresholds = np.concatenate([[thresholds[0] - 1e-6], thresholds, [thresholds[-1] + 1e-6]])

    far = np.array([(bona_fide_scores >= t).mean() for t in thresholds])   # falsa alarma
    frr = np.array([(spoof_scores < t).mean() for t in thresholds])        # falso rechazo/miss

    idx = np.nanargmin(np.abs(far - frr))
    eer = (far[idx] + frr[idx]) / 2
    return float(eer), float(thresholds[idx])


def classification_report_dict(y_true, y_pred) -> dict:
    """y_true/y_pred: 1 = sintético (spoof), 0 = real (bona fide)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
