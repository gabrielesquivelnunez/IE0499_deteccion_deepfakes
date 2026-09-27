"""
run_entregable2_experiments.py

Corre lo que pide explícitamente el entregable 2: experimentos con varias
duraciones de ventana (0.5, 1, 2, 3, 5 s) y una comparación básica entre
el Detector A (LFCC+SVM) y el Detector B (AASIST-L).

- Detector A: se entrena y evalúa con leave-one-speaker-out (LOSO), dado
  que el dataset real tiene pocos locutores. Se agrupan los puntajes de
  las 4 rondas (una por locutor excluido) para calcular un solo EER por
  duración de ventana.
- Detector B: se usa AASIST-L tal como viene preentrenado (sin fine-tuning
  todavía -- eso es trabajo del entregable 3), simplemente para confirmar
  que también puede recibir ventanas y producir un puntaje, y tener un
  punto de comparación inicial contra el Detector A.

Uso:
    python -m scripts.run_entregable2_experiments --raw-dir /ruta/a/voice-similarity-eval/data/raw

Genera results/entregable2_resultados.csv y lo imprime en pantalla.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.prepare_dataset import scan_dataset, leave_one_speaker_out_splits
from features.windowing import make_windows
from features.lfcc import lfcc_window_features
from metrics.metrics import compute_eer

WINDOW_SECS = [0.5, 1.0, 2.0, 3.0, 5.0]


def load_audio_mono(path: str):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def resample_to(audio: np.ndarray, sr_in: int, sr_out: int = 16000) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    gcd = np.gcd(sr_in, sr_out)
    return resample_poly(audio, sr_out // gcd, sr_in // gcd).astype(np.float32)


def run_detector_a_loso(entries, window_sec: float):
    """LFCC + SVM, leave-one-speaker-out. Devuelve (eer, n_bonafide, n_spoof)."""
    all_bonafide, all_spoof = [], []

    for held_out, train_entries, test_entries in leave_one_speaker_out_splits(entries):
        X_train, y_train = [], []
        for e in train_entries:
            audio, sr = load_audio_mono(e.path)
            label = 1 if e.label == "fake" else 0
            for w in make_windows(audio, sr, window_sec, window_sec):
                X_train.append(lfcc_window_features(w, sr))
                y_train.append(label)

        if len(set(y_train)) < 2:
            # esta ronda de LOSO no tiene ambas clases en train, no se puede entrenar
            continue

        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced"),
        )
        clf.fit(X_train, y_train)

        for e in test_entries:
            audio, sr = load_audio_mono(e.path)
            for w in make_windows(audio, sr, window_sec, window_sec):
                feat = lfcc_window_features(w, sr).reshape(1, -1)
                prob_fake = clf.predict_proba(feat)[0, 1]
                if e.label == "fake":
                    all_spoof.append(prob_fake)
                else:
                    all_bonafide.append(prob_fake)

    if not all_bonafide or not all_spoof:
        return None, len(all_bonafide), len(all_spoof)

    eer, _ = compute_eer(np.array(all_bonafide), np.array(all_spoof))
    return eer, len(all_bonafide), len(all_spoof)


def run_detector_b_zero_shot(entries, window_sec: float, detector):
    """AASIST-L preentrenado, sin fine-tuning. Devuelve (eer, n_bonafide, n_spoof)."""
    all_bonafide, all_spoof = [], []

    for e in entries:
        audio, sr = load_audio_mono(e.path)
        audio_16k = resample_to(audio, sr, 16000)
        for w in make_windows(audio_16k, 16000, window_sec, window_sec):
            score = detector.score_window(w, sr=16000)
            if e.label == "fake":
                all_spoof.append(score)
            else:
                all_bonafide.append(score)

    if not all_bonafide or not all_spoof:
        return None, len(all_bonafide), len(all_spoof)

    eer, _ = compute_eer(np.array(all_bonafide), np.array(all_spoof))
    return eer, len(all_bonafide), len(all_spoof)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True,
                         help="Carpeta con los .wav reales (ej. .../voice-similarity-eval/data/raw)")
    parser.add_argument("--skip-detector-b", action="store_true",
                         help="Omitir AASIST-L (por si PyTorch/GPU no está disponible)")
    parser.add_argument("--out", default="results/entregable2_resultados.csv")
    args = parser.parse_args()

    entries = scan_dataset(args.raw_dir)
    if not entries:
        print(f"No se encontraron .wav en {args.raw_dir}")
        return

    n_speakers = len({e.group_id for e in entries})
    n_real = sum(1 for e in entries if e.label == "real")
    n_fake = sum(1 for e in entries if e.label == "fake")
    print(f"Dataset: {len(entries)} archivos | {n_real} reales, {n_fake} sintéticos | {n_speakers} locutores\n")

    detector_b = None
    if not args.skip_detector_b:
        try:
            from models.detector_b_aasist import AASISTDetector
            detector_b = AASISTDetector()
        except Exception as exc:
            print(f"AVISO: no se pudo cargar AASIST-L ({exc}). Se omite el Detector B.\n")

    rows = []
    for window_sec in WINDOW_SECS:
        print(f"=== Ventana: {window_sec}s ===")

        t0 = time.time()
        eer_a, n_bf_a, n_sp_a = run_detector_a_loso(entries, window_sec)
        t_a = time.time() - t0
        eer_a_str = f"{eer_a*100:.2f}%" if eer_a is not None else "N/A"
        print(f"  Detector A (LFCC+SVM, LOSO): EER={eer_a_str}  "
              f"({n_bf_a} ventanas reales, {n_sp_a} sintéticas, {t_a:.1f}s)")
        rows.append({"window_sec": window_sec, "detector": "A (LFCC+SVM)",
                     "eer": eer_a, "n_bonafide": n_bf_a, "n_spoof": n_sp_a, "tiempo_s": round(t_a, 1)})

        if detector_b is not None:
            t0 = time.time()
            eer_b, n_bf_b, n_sp_b = run_detector_b_zero_shot(entries, window_sec, detector_b)
            t_b = time.time() - t0
            eer_b_str = f"{eer_b*100:.2f}%" if eer_b is not None else "N/A"
            print(f"  Detector B (AASIST-L, zero-shot): EER={eer_b_str}  "
                  f"({n_bf_b} ventanas reales, {n_sp_b} sintéticas, {t_b:.1f}s)")
            rows.append({"window_sec": window_sec, "detector": "B (AASIST-L, zero-shot)",
                         "eer": eer_b, "n_bonafide": n_bf_b, "n_spoof": n_sp_b, "tiempo_s": round(t_b, 1)})
        print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["window_sec", "detector", "eer", "n_bonafide", "n_spoof", "tiempo_s"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Resultados guardados en {args.out}")


if __name__ == "__main__":
    main()
