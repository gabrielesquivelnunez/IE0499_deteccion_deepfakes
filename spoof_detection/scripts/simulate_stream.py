"""
simulate_stream.py

Reproduce un archivo de audio como flujo continuo y muestra, ventana a
ventana, el puntaje del detector y la decisión de la estrategia temporal.

Uso (desde spoof_detection/):
    python -m scripts.simulate_stream --wav RUTA.wav --raw-dir RUTA/data/raw \
        --detector a --window-sec 2 --overlap 0.5 --strategy sostenido_k2

  --detector a : LFCC+SVM entrenado con los OTROS locutores (el del archivo se excluye)
  --detector b : AASIST-L (--finetuned no disponible aquí; ver run_finetune_loso.py)
  --realtime   : espera entre chunks como si el audio llegara en vivo
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.prepare_dataset import scan_dataset
from features.audio_io import load_audio_16k
from streaming.decision import default_strategies
from streaming.pipeline import StreamingDetector, stream_audio

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--detector", choices=["a", "b"], default="a")
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--strategy", default="sostenido_k2", choices=list(default_strategies()))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--chunk-sec", type=float, default=0.1)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--out", default=None, help="CSV opcional con la línea de tiempo")
    args = ap.parse_args()

    audio, sr = load_audio_16k(args.wav)
    hop_sec = args.window_sec * (1 - args.overlap)

    if args.detector == "a":
        from models.detector_a_stream import train_detector_a
        entries = scan_dataset(args.raw_dir)
        me = next((e for e in entries if Path(e.path).resolve() == Path(args.wav).resolve()), None)
        train = [e for e in entries if me is None or e.group_id != me.group_id]
        print(f"Entrenando Detector A con {len(train)} archivos"
              f"{'' if me is None else f' (excluido locutor {me.group_id})'}...")
        score_fn = train_detector_a(train, args.window_sec).score_window
    else:
        from models.detector_b_aasist import AASISTDetector
        score_fn = AASISTDetector().score_window

    decider = default_strategies(args.threshold)[args.strategy]()
    sd = StreamingDetector(score_fn, sr, args.window_sec, hop_sec, decider)

    print(f"\n{Path(args.wav).name}: {len(audio)/sr:.1f} s | ventana {args.window_sec} s, "
          f"salto {hop_sec:.2f} s | estrategia {args.strategy}\n")
    print(f"{'t(s)':>6s} {'puntaje':>8s} {'decisión':>10s} {'cómputo(ms)':>12s}")
    events = stream_audio(sd, audio, args.chunk_sec, args.realtime)
    for ev in events:
        dec = "—" if not ev.ready else ("FALSO" if ev.is_fake else "real")
        print(f"{ev.t_end:6.2f} {ev.score:8.3f} {dec:>10s} {ev.latency_s*1000:12.1f}")

    first = next((ev for ev in events if ev.ready), None)
    if first:
        print(f"\nPrimera decisión: {'FALSO' if first.is_fake else 'real'} a los {first.t_end:.2f} s de audio")
    else:
        print("\nNo se alcanzó una decisión (audio demasiado corto para la estrategia).")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["window_index", "t_end", "score", "ready", "is_fake", "latency_s"])
            for ev in events:
                w.writerow([ev.window_index, round(ev.t_end, 3), round(ev.score, 5),
                            ev.ready, ev.is_fake, round(ev.latency_s, 5)])


if __name__ == "__main__":
    main()
