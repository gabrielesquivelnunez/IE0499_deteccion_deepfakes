"""
run_entregable2_experiments.py  (versión corregida: todo el audio a 16 kHz)

Barrido de ventanas (0.5, 1, 2, 3, 5 s) comparando:
  - Detector A: LFCC + SVM, leave-one-speaker-out (LOSO)
  - Detector B: AASIST-L zero-shot

Todo el audio se remuestrea a 16 kHz ANTES de extraer características, para
que la tasa de muestreo no delate la clase (ver scripts/placebo_sample_rate.py).

Uso (desde spoof_detection/):
    python -m scripts.run_entregable2_experiments --raw-dir /ruta/a/data/raw
"""

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.prepare_dataset import scan_dataset, leave_one_speaker_out_splits
from features.audio_io import load_audio_16k
from features.lfcc import lfcc_window_features
from features.windowing import make_windows
from metrics.metrics import compute_eer

warnings.filterwarnings("ignore", category=FutureWarning)


def detector_a_scores(entries, window_sec):
    """Puntajes por ventana (sin solape) del Detector A con LOSO."""
    scores, labels = [], []
    for _held, train_entries, test_entries in leave_one_speaker_out_splits(entries):
        X, y = [], []
        for e in train_entries:
            audio, sr = load_audio_16k(e.path)
            for w in make_windows(audio, sr, window_sec, window_sec):
                X.append(lfcc_window_features(w, sr)); y.append(int(e.label == "fake"))
        if len(set(y)) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            SVC(kernel="rbf", C=1.0, gamma="scale",
                                probability=True, class_weight="balanced"))
        clf.fit(X, y)
        for e in test_entries:
            audio, sr = load_audio_16k(e.path)
            wins = make_windows(audio, sr, window_sec, window_sec)
            if not wins:
                continue
            p = clf.predict_proba(np.array([lfcc_window_features(w, sr) for w in wins]))[:, 1]
            scores += list(p); labels += [int(e.label == "fake")] * len(p)
    return np.array(scores), np.array(labels)


def detector_b_scores(entries, window_sec, detector):
    scores, labels = [], []
    for e in entries:
        audio, sr = load_audio_16k(e.path)
        for w in make_windows(audio, sr, window_sec, window_sec):
            scores.append(detector.score_window(w, sr=sr)); labels.append(int(e.label == "fake"))
    return np.array(scores), np.array(labels)


def eer_pct(scores, labels):
    eer, _ = compute_eer(bona_fide_scores=scores[labels == 0], spoof_scores=scores[labels == 1])
    return eer * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--window-secs", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 5.0])
    ap.add_argument("--no-detector-b", action="store_true")
    ap.add_argument("--out", default="results/entregable2_resultados.csv")
    args = ap.parse_args()

    entries = scan_dataset(args.raw_dir)
    n_fake = sum(e.label == "fake" for e in entries)
    print(f"Dataset: {len(entries)} archivos | {len(entries)-n_fake} reales, {n_fake} sintéticos "
          f"| {len({e.group_id for e in entries})} locutores | audio remuestreado a 16 kHz\n")

    detector_b = None
    if not args.no_detector_b:
        from models.detector_b_aasist import AASISTDetector
        detector_b = AASISTDetector()

    rows = []
    for ws in args.window_secs:
        print(f"=== Ventana: {ws}s ===")
        t0 = time.time()
        s, y = detector_a_scores(entries, ws)
        eer_a = eer_pct(s, y)
        print(f"  Detector A (LFCC+SVM, LOSO): EER={eer_a:.2f}%  "
              f"({(y==0).sum()} ventanas reales, {(y==1).sum()} sintéticas, {time.time()-t0:.1f}s)")
        row = {"window_sec": ws, "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
               "eer_detector_a": round(eer_a, 2)}
        if detector_b is not None:
            t0 = time.time()
            sb, yb = detector_b_scores(entries, ws, detector_b)
            eer_b = eer_pct(sb, yb)
            print(f"  Detector B (AASIST-L, zero-shot): EER={eer_b:.2f}%  ({time.time()-t0:.1f}s)")
            row["eer_detector_b"] = round(eer_b, 2)
        rows.append(row); print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Resultados guardados en {args.out}")


if __name__ == "__main__":
    main()
