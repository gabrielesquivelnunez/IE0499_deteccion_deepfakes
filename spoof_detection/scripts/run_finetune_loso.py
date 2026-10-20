"""
run_finetune_loso.py

AASIST-L con fine-tuning evaluado con leave-one-speaker-out (LOSO):
para cada locutor retenido se parte de los pesos preentrenados, se ajusta con
los OTROS locutores y se puntúa al retenido. Se reporta, por ventana:
  - EER zero-shot (sin ajuste) y EER con fine-tuning
  - métricas por estrategia temporal (igual que run_temporal_strategies.py)

Uso (desde spoof_detection/):
    python -m scripts.run_finetune_loso --raw-dir RUTA/data/raw
Salidas: results/entregable3_b_finetune_eer.csv y
         results/entregable3_b_finetune_estrategias.csv
"""

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.prepare_dataset import scan_dataset, leave_one_speaker_out_splits
from features.audio_io import load_audio_16k
from features.windowing import make_windows
from metrics.metrics import compute_eer
from models.detector_b_aasist import AASISTDetector
from models.finetune_aasist import finetune, make_training_windows
from streaming.decision import default_strategies, summarize_strategy

warnings.filterwarnings("ignore")


def score_stream(det, entry, window_sec, hop_sec):
    audio, sr = load_audio_16k(entry.path)
    wins = make_windows(audio, sr, window_sec, hop_sec)
    if not wins:
        return None
    return np.array([det.score_window(w, sr) for w in wins])


def eer_pct(streams):
    s = np.concatenate([x for x, _ in streams]); y = np.concatenate([[l] * len(x) for x, l in streams])
    eer, _ = compute_eer(bona_fide_scores=s[y == 0], spoof_scores=s[y == 1])
    return eer * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--window-secs", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--overlaps", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-prefix", default="results/entregable3_b_finetune")
    args = ap.parse_args()

    entries = scan_dataset(args.raw_dir)
    if not entries:
        sys.exit(f"No se encontraron .wav en {args.raw_dir}. Revisa la ruta --raw-dir.")
    det = AASISTDetector(device=args.device)
    strategies = default_strategies(args.threshold)
    eer_rows, strat_rows = [], []

    for ws in args.window_secs:
        zero, tuned = {ov: [] for ov in args.overlaps}, {ov: [] for ov in args.overlaps}
        t0 = time.time()
        for held, train_entries, test_entries in leave_one_speaker_out_splits(entries):
            print(f"[ventana {ws}s] locutor retenido: {held}")
            labels = {e.label for e in train_entries}
            if labels != {"real", "fake"}:
                continue
            det.reset_weights()
            for ov in args.overlaps:   # zero-shot
                for e in test_entries:
                    s = score_stream(det, e, ws, ws * (1 - ov))
                    if s is not None:
                        zero[ov].append((s, int(e.label == "fake")))
            finetune(det, make_training_windows(train_entries, ws), epochs=args.epochs, lr=args.lr)
            for ov in args.overlaps:
                for e in test_entries:
                    s = score_stream(det, e, ws, ws * (1 - ov))
                    if s is not None:
                        tuned[ov].append((s, int(e.label == "fake")))

        eer_z, eer_t = eer_pct(zero[0.0]), eer_pct(tuned[0.0])
        print(f"== ventana {ws}s: EER zero-shot {eer_z:.2f}%  |  con fine-tuning {eer_t:.2f}%  "
              f"({time.time()-t0:.0f}s)\n")
        eer_rows.append({"window_sec": ws, "eer_zero_shot": round(eer_z, 2),
                         "eer_finetuned": round(eer_t, 2), "epochs": args.epochs, "lr": args.lr})

        for ov in args.overlaps:
            hop = ws * (1 - ov)
            for name, make in strategies.items():
                m = summarize_strategy(tuned[ov], make, ws, hop)
                strat_rows.append({"detector": "B (AASIST-L fine-tuned, LOSO)", "window_sec": ws,
                                   "overlap": ov, "hop_sec": round(hop, 3), "strategy": name,
                                   "threshold": args.threshold, **m})

    for suffix, rows in (("_eer.csv", eer_rows), ("_estrategias.csv", strat_rows)):
        path = Path(args.out_prefix + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("Guardado:", path)


if __name__ == "__main__":
    main()
