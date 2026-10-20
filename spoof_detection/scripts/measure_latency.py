"""
measure_latency.py

Latencia de inferencia por ventana y Real-Time Factor (RTF) de los detectores.

  latencia = tiempo de cómputo para puntuar UNA ventana (extraer + clasificar)
  RTF      = latencia / segundos de audio que avanza el flujo por ventana (el salto)
             RTF < 1  => el sistema sigue el ritmo del audio en vivo.
  Se reporta para solape 0 % (salto = ventana) y 50 % (salto = ventana/2).

Uso (desde spoof_detection/):
    python -m scripts.measure_latency --raw-dir RUTA/data/raw
"""

import argparse
import csv
import platform
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.prepare_dataset import scan_dataset
from features.audio_io import load_audio_16k
from features.windowing import make_windows

warnings.filterwarnings("ignore")


def bench(fn, windows, reps, warmup=5, sync=None):
    for i in range(warmup):
        fn(windows[i % len(windows)])
    if sync: sync()
    times, cpu0 = [], time.process_time()
    for i in range(reps):
        w = windows[i % len(windows)]
        t0 = time.perf_counter()
        fn(w)
        if sync: sync()
        times.append(time.perf_counter() - t0)
    cpu_frac = (time.process_time() - cpu0) / max(sum(times), 1e-9)
    t = np.array(times)
    return t.mean(), np.median(t), np.percentile(t, 95), cpu_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--window-secs", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 5.0])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--no-b", action="store_true")
    ap.add_argument("--out", default="results/entregable3_latencia_rtf.csv")
    args = ap.parse_args()

    entries = scan_dataset(args.raw_dir)
    if not entries:
        sys.exit(f"No se encontraron .wav en {args.raw_dir}. Revisa la ruta --raw-dir.")
    real = [e for e in entries if e.label == "real"]
    audio, sr = load_audio_16k(real[0].path)

    from models.detector_a_stream import train_detector_a
    devices = []
    if not args.no_b:
        import torch
        from models.detector_b_aasist import AASISTDetector
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        dets_b = {d: AASISTDetector(device=d, verbose=False) for d in devices}
        n_params = dets_b["cpu"].n_parameters()

    print(f"CPU: {platform.processor() or platform.machine()} | python {platform.python_version()}")
    rows = []
    for ws in args.window_secs:
        wins = make_windows(audio, sr, ws, ws) or [np.resize(audio, int(ws * sr))]
        cases = {}
        a = train_detector_a(entries, ws)
        cases["A (LFCC+SVM) CPU"] = (lambda w, a=a: a.score_window(w, sr), None)
        for d in devices:
            det = dets_b[d]
            sync = (lambda: torch.cuda.synchronize()) if d == "cuda" else None
            cases[f"B (AASIST-L) {d.upper()}"] = (lambda w, det=det: det.score_window(w, sr), sync)

        for name, (fn, sync) in cases.items():
            mean, med, p95, cpu = bench(fn, wins, args.reps, sync=sync)
            row = {"detector": name, "window_sec": ws, "lat_mean_ms": round(mean*1e3, 2),
                   "lat_median_ms": round(med*1e3, 2), "lat_p95_ms": round(p95*1e3, 2),
                   "rtf_overlap0": round(mean/ws, 4), "rtf_overlap50": round(mean/(ws/2), 4),
                   "cpu_busy_ratio": round(cpu, 2)}
            rows.append(row)
            print(f"{name:20s} vent {ws:3.1f}s  lat {mean*1e3:7.1f} ms (mediana {med*1e3:6.1f}, "
                  f"p95 {p95*1e3:6.1f})  RTF0={row['rtf_overlap0']:.3f}  RTF50={row['rtf_overlap50']:.3f}")

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"\nMemoria RAM máxima del proceso: {rss_mb:.0f} MB")
    if devices:
        print(f"Parámetros AASIST-L: {n_params:,}")
        if "cuda" in devices:
            print(f"Memoria GPU máxima: {torch.cuda.max_memory_allocated()/2**20:.0f} MB")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Guardado en {args.out}")


if __name__ == "__main__":
    main()
