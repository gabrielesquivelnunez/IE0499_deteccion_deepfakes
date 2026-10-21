"""
run_finetune_loso.py

AASIST-L con ajuste fino evaluado con leave-one-speaker-out (LOSO): para cada
locutor retenido se parte de los pesos preentrenados, se ajusta con los demás
locutores y se puntúa al retenido.

Por cada duración de ventana se escriben, en results/:
  <prefijo>_eer_<w>s.csv           EER sin ajuste y con ajuste
  <prefijo>_estrategias_<w>s.csv   métricas por estrategia temporal, con umbral fijo
                                   y con umbral calibrado
  puntajes_B_<w>s.json             puntajes por ventana de cada flujo

El umbral calibrado de cada locutor retenido es el punto de EER de los puntajes
de los demás locutores, por lo que no usa información del locutor evaluado.

Uso (desde spoof_detection/):
    python -m scripts.run_finetune_loso --raw-dir RUTA/data/raw --window-secs 1 2 4
"""

import argparse
import csv
import json
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

EPS = 1e-4


def score_stream(det, entry, window_sec, hop_sec):
    audio, sr = load_audio_16k(entry.path)
    wins = make_windows(audio, sr, window_sec, hop_sec)
    if not wins:
        return None
    return np.array([det.score_window(w, sr) for w in wins])


def pooled(streams):
    s = np.concatenate([d["scores"] for d in streams])
    y = np.concatenate([[d["label"]] * len(d["scores"]) for d in streams])
    return s, y


def eer_pct(streams):
    s, y = pooled(streams)
    eer, _ = compute_eer(bona_fide_scores=s[y == 0], spoof_scores=s[y == 1])
    return eer * 100


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibrate(streams):
    """Desplaza los puntajes de cada flujo con el umbral de EER de los demás
    locutores, de modo que el umbral de decisión pase a ser 0.5.
    Devuelve la lista de flujos (puntajes, etiqueta) y el umbral de cada locutor."""
    out, thresholds = [], {}
    for spk in sorted({d["spk"] for d in streams}):
        rest = [d for d in streams if d["spk"] != spk]
        thr = 0.5
        if len({d["label"] for d in rest}) == 2:
            s, y = pooled(rest)
            _, thr = compute_eer(bona_fide_scores=s[y == 0], spoof_scores=s[y == 1])
        thr = float(np.clip(thr, EPS, 1 - EPS))
        thresholds[spk] = thr
        for d in streams:
            if d["spk"] == spk:
                out.append((1 / (1 + np.exp(-(logit(d["scores"]) - logit(thr)))), d["label"]))
    return out, thresholds


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Guardado:", path)


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
        sys.exit(f"No se encontraron archivos .wav en {args.raw_dir}")
    det = AASISTDetector(device=args.device)
    strategies = default_strategies(args.threshold)
    print(f"Dataset: {len(entries)} archivos | "
          f"{len({e.group_id for e in entries})} locutores | ventanas {args.window_secs} s")

    for ws in args.window_secs:
        tag = f"{ws:g}s"
        recs = {ov: [] for ov in args.overlaps}
        t0 = time.time()
        for held, train_entries, test_entries in leave_one_speaker_out_splits(entries):
            print(f"[ventana {ws:g}s] locutor retenido: {held}")
            if {e.label for e in train_entries} != {"real", "fake"}:
                continue
            det.reset_weights()
            zero = {}
            for ov in args.overlaps:
                for e in test_entries:
                    s = score_stream(det, e, ws, ws * (1 - ov))
                    if s is not None:
                        zero[(ov, e.path)] = s
            finetune(det, make_training_windows(train_entries, ws), epochs=args.epochs, lr=args.lr)
            for ov in args.overlaps:
                for e in test_entries:
                    s = score_stream(det, e, ws, ws * (1 - ov))
                    if s is not None:
                        recs[ov].append({"spk": held, "name": Path(e.path).name,
                                         "label": int(e.label == "fake"),
                                         "zero": zero[(ov, e.path)], "scores": s})

        base = recs[0.0] if 0.0 in recs else recs[args.overlaps[0]]
        eer_z = eer_pct([{**d, "scores": d["zero"]} for d in base])
        eer_t = eer_pct(base)
        _, thr_by_spk = calibrate(base)
        print(f"== ventana {ws:g}s: EER sin ajuste {eer_z:.2f}% | con ajuste {eer_t:.2f}% | "
              f"umbral calibrado medio {np.mean(list(thr_by_spk.values())):.3f} ({time.time()-t0:.0f}s)\n")
        write_csv(f"{args.out_prefix}_eer_{tag}.csv", [{
            "window_sec": ws, "eer_zero_shot": round(eer_z, 2), "eer_finetuned": round(eer_t, 2),
            "epochs": args.epochs, "lr": args.lr,
            "umbral_calibrado_medio": round(float(np.mean(list(thr_by_spk.values()))), 3)}])

        strat_rows = []
        for ov in args.overlaps:
            hop = ws * (1 - ov)
            fixed = [(d["scores"], d["label"]) for d in recs[ov]]
            calibrated, thr = calibrate(recs[ov])
            for mode, streams, thr_value in (("fijo", fixed, args.threshold),
                                             ("calibrado", calibrated, float(np.mean(list(thr.values()))))):
                for name, make in strategies.items():
                    m = summarize_strategy(streams, make, ws, hop)
                    strat_rows.append({"detector": "B (AASIST-L ajustado, LOSO)", "window_sec": ws,
                                       "overlap": ov, "hop_sec": round(hop, 3), "strategy": name,
                                       "umbral_modo": mode, "threshold": round(thr_value, 4), **m})
        write_csv(f"{args.out_prefix}_estrategias_{tag}.csv", strat_rows)

        scores_path = Path(args.out_prefix).parent / f"puntajes_B_{tag}.json"
        with open(scores_path, "w") as f:
            json.dump({"window_sec": ws, "flujos": [
                {"locutor": d["spk"], "archivo": d["name"], "etiqueta": d["label"], "solape": ov,
                 "sin_ajuste": [round(float(x), 5) for x in d["zero"]],
                 "ajustado": [round(float(x), 5) for x in d["scores"]]}
                for ov in args.overlaps for d in recs[ov]]}, f)
        print("Guardado:", scores_path)


if __name__ == "__main__":
    main()
