"""
placebo_sample_rate.py

Prueba PLACEBO de la confusión tasa de muestreo / etiqueta.

Se genera audio sintético-artificial donde "real" y "falso" salen EXACTAMENTE
del mismo proceso aleatorio (no hay ninguna diferencia real entre clases). Lo
único que difiere es la tasa de muestreo: "real" se guarda a 16 kHz y "falso" a
24 kHz, igual que en el dataset real. Un detector honesto debe dar EER ~ 50 %.

  - Pipeline VIEJO (LFCC a la tasa original): si da EER ~ 0 %, detecta la tasa.
  - Pipeline NUEVO (remuestrear a 16 kHz): debe dar EER ~ 50 %.

Uso (desde spoof_detection/):  python -m scripts.placebo_sample_rate
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.audio_io import to_mono_16k, resample_to
from features.lfcc import lfcc_window_features
from features.windowing import make_windows
from metrics.metrics import compute_eer

warnings.filterwarnings("ignore")
HI = 24000


def make_signal(rng, seconds=8.0):
    """Tipo voz: tren de pulsos (f0 variable) filtrado + ruido, banda < 7 kHz,
    generado a 24 kHz."""
    n = int(seconds * HI)
    f0 = 110 + 25 * np.sin(2 * np.pi * rng.uniform(0.2, 0.6) * np.arange(n) / HI)
    phase = np.cumsum(f0) / HI
    src = (np.diff(np.floor(phase), prepend=0) > 0).astype(float)
    x = src + 0.05 * rng.standard_normal(n)
    sos = butter(4, [150, 7000], btype="band", fs=HI, output="sos")
    x = sosfilt(sos, x)
    env = 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(2, 4) * np.arange(n) / HI) ** 2
    return (x * env / np.abs(x).max()).astype(np.float32)


def build_files(n_per_class=24, seed=0):
    rng = np.random.default_rng(seed)
    files = []
    for i in range(2 * n_per_class):
        sig = make_signal(rng)
        label = i % 2              # 0 = "real", 1 = "falso": asignación arbitraria
        spk = i // 4 % 6           # 6 "locutores" ficticios (para LOSO)
        if label == 0:
            files.append((resample_to(sig, HI, 16000), 16000, label, spk))  # a 16k
        else:
            files.append((sig, HI, label, spk))                             # queda a 24k
    return files


def featurize(files, window_sec, fix):
    X, y, g = [], [], []
    for audio, sr, label, spk in files:
        if fix:
            audio, sr = to_mono_16k(audio, sr), 16000
        for w in make_windows(audio, sr, window_sec, window_sec):
            X.append(lfcc_window_features(w, sr)); y.append(label); g.append(spk)
    return np.array(X), np.array(y), np.array(g)


def loso_eer(X, y, g):
    scores = np.zeros(len(y))
    for s in np.unique(g):
        tr, te = g != s, g == s
        clf = make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        scores[te] = clf.predict_proba(X[te])[:, 1]
    eer, _ = compute_eer(bona_fide_scores=scores[y == 0], spoof_scores=scores[y == 1])
    return eer * 100


def main():
    files = build_files()
    print("PLACEBO: clases idénticas salvo tasa de muestreo (real=16 kHz, falso=24 kHz)")
    print("EER esperado de un detector honesto: ~50 %\n")
    print(f"{'ventana':>8s} {'pipeline VIEJO':>16s} {'pipeline NUEVO (16k)':>22s}")
    for ws in (0.5, 1.0, 2.0):
        old = loso_eer(*featurize(files, ws, fix=False))
        new = loso_eer(*featurize(files, ws, fix=True))
        print(f"{ws:7.1f}s {old:15.2f}% {new:21.2f}%")


if __name__ == "__main__":
    main()
