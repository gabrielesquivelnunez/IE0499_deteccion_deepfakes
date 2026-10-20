"""Detector A listo para streaming: entrena LFCC+SVM y expone score_window()."""

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features.audio_io import load_audio_16k
from features.lfcc import lfcc_window_features
from features.windowing import make_windows


class LFCCSVMDetector:
    def __init__(self, clf):
        self.clf = clf

    def score_window(self, audio: np.ndarray, sr: int) -> float:
        feat = lfcc_window_features(audio, sr).reshape(1, -1)
        return float(self.clf.predict_proba(feat)[0, 1])

    def __call__(self, audio, sr):
        return self.score_window(audio, sr)


def train_detector_a(train_entries, window_sec: float) -> LFCCSVMDetector:
    X, y = [], []
    for e in train_entries:
        audio, sr = load_audio_16k(e.path)
        for w in make_windows(audio, sr, window_sec, window_sec):
            X.append(lfcc_window_features(w, sr))
            y.append(int(e.label == "fake"))
    if len(set(y)) < 2:
        raise ValueError("el entrenamiento necesita ventanas reales y sintéticas")
    clf = make_pipeline(StandardScaler(),
                        SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                            class_weight="balanced", random_state=0))
    clf.fit(np.array(X), np.array(y))
    return LFCCSVMDetector(clf)
