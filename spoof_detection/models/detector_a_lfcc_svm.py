"""
detector_a_lfcc_svm.py

Detector A del proyecto: características LFCC + clasificador SVM.

Uso:
    python -m models.detector_a_lfcc_svm \\
        --train data/splits/train.csv \\
        --val data/splits/val.csv \\
        --test data/splits/test.csv \\
        --window-sec 1.0 --hop-sec 1.0

Qué hace:
    1. Por cada archivo de audio en la partición, lo corta en ventanas
       (features.windowing.make_windows).
    2. Por cada ventana, extrae un vector LFCC de tamaño fijo
       (features.lfcc.lfcc_window_features).
    3. Entrena un SVM (kernel RBF, con probabilidades) sobre las ventanas
       de train.
    4. Evalúa en val/test: EER, accuracy, precision, recall, F1.

Nota: cada ventana hereda la etiqueta (real/sintético) del archivo del que
proviene. Es una simplificación razonable para esta primera fase del
proyecto (detector base); en el entregable 3 se estudiará si conviene una
estrategia distinta.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.windowing import make_windows
from features.lfcc import lfcc_window_features
from metrics.metrics import compute_eer, classification_report_dict


def load_split(csv_path: str) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def build_feature_matrix(entries: list[dict], window_sec: float, hop_sec: float,
                          verbose: bool = True):
    X, y, meta = [], [], []
    for i, entry in enumerate(entries):
        audio, sr = sf.read(entry["path"])
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # a mono si viene estéreo

        windows = make_windows(audio, sr, window_sec, hop_sec)
        label = 1 if entry["label"] == "fake" else 0

        for w in windows:
            feat = lfcc_window_features(w, sr)
            X.append(feat)
            y.append(label)
            meta.append(entry["path"])

        if verbose and (i + 1) % 50 == 0:
            print(f"  procesados {i + 1}/{len(entries)} archivos...")

    return np.array(X), np.array(y), meta


def train_svm(X_train, y_train):
    clf = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
            ensemble=False,
        ),
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, X, y, split_name: str):
    probs = clf.predict_proba(X)[:, 1]  # probabilidad de "fake"
    preds = (probs >= 0.5).astype(int)

    eer, threshold = compute_eer(
        bona_fide_scores=probs[y == 0],
        spoof_scores=probs[y == 1],
    )
    report = classification_report_dict(y, preds)

    print(f"\n--- Resultados en {split_name} ---")
    print(f"  EER: {eer * 100:.2f}%  (umbral: {threshold:.3f})")
    for k, v in report.items():
        print(f"  {k}: {v:.3f}")
    return {"eer": eer, "threshold": threshold, **report}


def main():
    parser = argparse.ArgumentParser(description="Entrena y evalúa el Detector A (LFCC + SVM)")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--hop-sec", type=float, default=None)
    parser.add_argument("--model-out", default="models/checkpoints/detector_a_lfcc_svm.joblib")
    args = parser.parse_args()

    hop_sec = args.hop_sec or args.window_sec

    print(f"Ventana: {args.window_sec}s | hop: {hop_sec}s")

    t0 = time.time()
    print("\nExtrayendo características de TRAIN...")
    X_train, y_train, _ = build_feature_matrix(load_split(args.train), args.window_sec, hop_sec)
    print(f"  {X_train.shape[0]} ventanas, {X_train.shape[1]} características cada una")

    print("\nEntrenando SVM...")
    clf = train_svm(X_train, y_train)
    print(f"  entrenamiento completo en {time.time() - t0:.1f}s")

    print("\nExtrayendo características de VAL...")
    X_val, y_val, _ = build_feature_matrix(load_split(args.val), args.window_sec, hop_sec)
    evaluate(clf, X_val, y_val, "validación")

    print("\nExtrayendo características de TEST...")
    X_test, y_test, _ = build_feature_matrix(load_split(args.test), args.window_sec, hop_sec)
    evaluate(clf, X_test, y_test, "prueba (test)")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.model_out)
    print(f"\nModelo guardado en {args.model_out}")


if __name__ == "__main__":
    main()
