"""
live_mic.py

Detección en vivo: captura audio del micrófono, lo pasa por el mismo pipeline
de streaming que simulate_stream.py (buffer -> ventanas -> detector ->
estrategia temporal) y muestra la decisión real/falso mientras se habla.

Uso (desde spoof_detection/, con el venv activo):
    python -m scripts.live_mic --list-devices
    python -m scripts.live_mic --raw-dir $RAW --detector a --window-sec 2 \
        --overlap 0.5 --strategy sostenido_k2 --seconds 30 --save-wav results/vivo.wav

  --detector a : LFCC+SVM entrenado con todo el conjunto de datos
  --detector b : AASIST-L
  --seconds 0  : corre hasta Ctrl+C
  --device N   : índice del micrófono (ver --list-devices)

Requiere:  pip install sounddevice   y   sudo apt install libportaudio2
"""

import argparse
import csv
import queue
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.audio_io import TARGET_SR, resample_to
from streaming.decision import default_strategies
from streaming.pipeline import StreamingDetector

warnings.filterwarnings("ignore")


def consume(q, detector, seconds, on_event, on_chunk=None, idle_timeout=2.0):
    """Saca chunks de la cola `q`, los empuja al pipeline y llama on_event(ev).
    Termina tras `seconds` de audio (0 = hasta KeyboardInterrupt) o si no llega
    audio durante `idle_timeout` s. Devuelve el total de muestras procesadas."""
    total = 0
    limit = int(seconds * detector.sr) if seconds > 0 else None
    while limit is None or total < limit:
        try:
            chunk = q.get(timeout=idle_timeout)
        except queue.Empty:
            print("\n(sin audio del micrófono; se detiene)")
            break
        total += len(chunk)
        if on_chunk:
            on_chunk(chunk)
        for ev in detector.push(chunk):
            on_event(ev, q.qsize())
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--raw-dir")
    ap.add_argument("--detector", choices=["a", "b"], default="a")
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--strategy", default="sostenido_k2", choices=list(default_strategies()))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--chunk-sec", type=float, default=0.1)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--save-wav", default=None, help="guarda lo grabado (16 kHz mono)")
    ap.add_argument("--out", default=None, help="CSV con la línea de tiempo")
    args = ap.parse_args()

    try:
        import sounddevice as sd
    except Exception as e:
        sys.exit(f"No se pudo importar sounddevice ({e}).\n"
                 "Instala:  pip install sounddevice   y   sudo apt install libportaudio2")

    if args.list_devices:
        print(sd.query_devices())
        return
    if not args.raw_dir:
        sys.exit("Falta --raw-dir")

    # ---- Detector ----
    if args.detector == "a":
        from data.prepare_dataset import scan_dataset
        from models.detector_a_stream import train_detector_a
        entries = scan_dataset(args.raw_dir)
        print(f"Entrenando Detector A con {len(entries)} archivos...")
        score_fn = train_detector_a(entries, args.window_sec).score_window
    else:
        from models.detector_b_aasist import AASISTDetector
        score_fn = AASISTDetector().score_window

    hop_sec = args.window_sec * (1 - args.overlap)
    decider = default_strategies(args.threshold)[args.strategy]()
    det = StreamingDetector(score_fn, TARGET_SR, args.window_sec, hop_sec, decider)

    # ---- Micrófono: se intenta abrir directo a 16 kHz; si no, tasa nativa + remuestreo ----
    q = queue.Queue()
    native_sr = int(sd.query_devices(args.device, "input")["default_samplerate"])
    stream, in_sr = None, TARGET_SR
    for sr_try in (TARGET_SR, native_sr):
        try:
            blk = int(sr_try * args.chunk_sec)
            stream = sd.InputStream(
                samplerate=sr_try, channels=1, dtype="float32", blocksize=blk, device=args.device,
                callback=lambda indata, frames, t, status: q.put(
                    resample_to(indata[:, 0].copy(), sr_try, TARGET_SR)))
            in_sr = sr_try
            break
        except Exception as e:
            print(f"No se pudo abrir el micrófono a {sr_try} Hz: {e}")
    if stream is None:
        sys.exit("No se pudo abrir ningún micrófono.")
    if in_sr != TARGET_SR:
        print(f"Aviso: micrófono a {in_sr} Hz; se remuestrea cada chunk a 16 kHz.")

    rec, events = [], []
    keep = lambda c: rec.append(c) if args.save_wav else None

    def on_event(ev, backlog):
        events.append(ev)
        dec = "…" if not ev.ready else ("FALSO" if ev.is_fake else "real")
        bar = "#" * int(round(ev.score * 20))
        warn = "  [!] atrasado" if ev.latency_s > hop_sec or backlog > 20 else ""
        print(f"{ev.t_end:6.1f}s  p={ev.score:5.3f} |{bar:<20s}| {dec:>6s}  "
              f"{ev.latency_s*1000:6.1f} ms{warn}", flush=True)

    print(f"\nEscuchando ({'hasta Ctrl+C' if args.seconds <= 0 else f'{args.seconds:.0f} s'}) | "
          f"ventana {args.window_sec} s, salto {hop_sec:.2f} s | {args.strategy}\n"
          "Habla (o reproduce un audio frente al micrófono)...\n")
    t_start = time.perf_counter()
    try:
        with stream:
            consume(q, det, args.seconds, on_event, keep)
    except KeyboardInterrupt:
        print("\nDetenido por el usuario.")

    # ---- Resumen ----
    if events:
        lat = np.array([e.latency_s for e in events])
        first = next((e for e in events if e.ready), None)
        n_fake = sum(1 for e in events if e.ready and e.is_fake)
        n_ready = sum(1 for e in events if e.ready)
        print(f"\nVentanas: {len(events)} | cómputo medio {lat.mean()*1000:.1f} ms "
              f"(máx {lat.max()*1000:.1f}) | RTF {lat.mean()/hop_sec:.3f}")
        if first:
            print(f"Primera decisión: {'FALSO' if first.is_fake else 'real'} a los {first.t_end:.1f} s de audio")
            print(f"Decisiones 'falso': {n_fake}/{n_ready}")
    else:
        print("No se completó ninguna ventana (grabación más corta que la ventana).")

    if args.save_wav and rec:
        import soundfile as sf
        Path(args.save_wav).parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.save_wav, np.concatenate(rec), TARGET_SR)
        print(f"Audio guardado en {args.save_wav}")
    if args.out and events:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["window_index", "t_end", "score", "ready", "is_fake", "latency_s"])
            for e in events:
                w.writerow([e.window_index, round(e.t_end, 3), round(e.score, 5),
                            e.ready, e.is_fake, round(e.latency_s, 5)])
        print(f"Línea de tiempo en {args.out}")


if __name__ == "__main__":
    main()
