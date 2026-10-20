"""
run_temporal_strategies.py

Entregable 3: compara estrategias de decisión temporal (ventana independiente,
promedio de probabilidades, voto mayoritario, umbral sostenido) variando la
duración de la ventana y el solapamiento.

Para cada configuración (ventana, solapamiento):
  1. Se obtiene, por cada archivo de audio, la SECUENCIA de puntajes del
     detector sobre ventanas sucesivas (como llegarían en un flujo continuo).
     - Detector A (LFCC+SVM): entrenado con leave-one-speaker-out; los
       puntajes de cada archivo vienen del modelo que NO vio a ese locutor.
     - Detector B (AASIST-L), opcional con --with-detector-b. Mientras no
       tenga fine-tuning (zero-shot), sus puntajes no son informativos y el
       resultado no es interpretable; usarlo cuando exista el modelo ajustado.
  2. Cada estrategia de streaming/decision.py procesa esa secuencia y se
     miden: cobertura, accuracy/precision/recall/F1 de la primera decisión,
     tiempo hasta la primera decisión, tasa de falsas alarmas y de detección.

Uso (desde spoof_detection/):
    python -m scripts.run_temporal_strategies --raw-dir /ruta/a/data/raw

Genera results/entregable3_estrategias_temporales.csv
"""

import argparse
import csv
import sys
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
from streaming.decision import default_strategies, summarize_strategy

warnings.filterwarnings("ignore", category=FutureWarning)  # aviso de SVC(probability=True)


def streams_detector_a_loso(entries, window_sec, hop_sec):
    """Secuencias de puntajes del Detector A. Entrena con ventanas sin solape
    (igual que en el entregable 2) y puntúa cada archivo de prueba con el
    paso de streaming (hop_sec). Devuelve lista de (scores, label, path)."""
    streams = []
    for _held_out, train_entries, test_entries in leave_one_speaker_out_splits(entries):
        X, y = [], []
        for e in train_entries:
            audio, sr = load_audio_16k(e.path)
            for w in make_windows(audio, sr, window_sec, window_sec):
                X.append(lfcc_window_features(w, sr))
                y.append(1 if e.label == "fake" else 0)
        if len(set(y)) < 2:
            continue

        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=0),
        )
        clf.fit(X, y)

        for e in test_entries:
            audio, sr = load_audio_16k(e.path)
            wins = make_windows(audio, sr, window_sec, hop_sec)
            if not wins:
                continue
            feats = np.array([lfcc_window_features(w, sr) for w in wins])
            probs = clf.predict_proba(feats)[:, 1]
            streams.append((probs, 1 if e.label == "fake" else 0, e.path))
    return streams


def streams_detector_b(entries, window_sec, hop_sec, detector):
    streams = []
    for e in entries:
        audio, sr = load_audio_16k(e.path)
        wins = make_windows(audio, sr, window_sec, hop_sec)
        if not wins:
            continue
        scores = np.array([detector.score_window(w, sr=sr) for w in wins])
        streams.append((scores, 1 if e.label == "fake" else 0, e.path))
    return streams


def fmt(x, pct=False, nd=3):
    if x != x:  # NaN
        return "  n/a"
    return f"{x*100:5.1f}%" if pct else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--window-secs", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--overlaps", type=float, nargs="+", default=[0.0, 0.5],
                    help="Fracción de solapamiento entre ventanas (0.5 = 50%%)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--with-detector-b", action="store_true")
    ap.add_argument("--out", default="results/entregable3_estrategias_temporales.csv")
    args = ap.parse_args()

    entries = scan_dataset(args.raw_dir)
    if not entries:
        print(f"No se encontraron .wav en {args.raw_dir}")
        return
    print(f"Dataset: {len(entries)} archivos, {len({e.group_id for e in entries})} locutores\n")

    detector_b = None
    if args.with_detector_b:
        from models.detector_b_aasist import AASISTDetector
        detector_b = AASISTDetector()

    strategies = default_strategies(args.threshold)
    rows = []

    for window_sec in args.window_secs:
        for overlap in args.overlaps:
            if not 0.0 <= overlap < 1.0:
                raise ValueError("el solapamiento debe estar en [0, 1)")
            hop_sec = window_sec * (1.0 - overlap)

            detectors = {"A (LFCC+SVM, LOSO)": streams_detector_a_loso(entries, window_sec, hop_sec)}
            if detector_b is not None:
                detectors["B (AASIST-L)"] = streams_detector_b(entries, window_sec, hop_sec, detector_b)

            for det_name, streams in detectors.items():
                pairs = [(s, y) for s, y, _ in streams]
                print(f"=== {det_name} | ventana {window_sec} s | solape {overlap:.0%} "
                      f"(salto {hop_sec:.2f} s) | {len(pairs)} flujos ===")
                print(f"{'estrategia':24s} {'cobert':>7s} {'acc':>7s} {'prec':>7s} {'recall':>7s} "
                      f"{'f1':>6s} {'t_1ra(s)':>8s} {'alarma':>7s} {'detec':>7s}")
                for name, make in strategies.items():
                    m = summarize_strategy(pairs, make, window_sec, hop_sec)
                    print(f"{name:24s} {fmt(m['coverage'], True):>7s} {fmt(m['accuracy'], True):>7s} "
                          f"{fmt(m['precision'], True):>7s} {fmt(m['recall'], True):>7s} "
                          f"{fmt(m['f1']):>6s} {fmt(m['mean_t_first'], nd=2):>8s} "
                          f"{fmt(m['false_alarm_rate'], True):>7s} {fmt(m['detect_rate'], True):>7s}")
                    rows.append({"detector": det_name, "window_sec": window_sec,
                                 "overlap": overlap, "hop_sec": round(hop_sec, 3),
                                 "strategy": name, "threshold": args.threshold, **m})
                print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Resultados guardados en {args.out}")


if __name__ == "__main__":
    main()
