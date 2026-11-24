"""
demo.py

Demostrador del detector de voces sintéticas (Detector A: LFCC + SVM) en flujo
continuo. Muestra, mientras llega el audio, la evolución del puntaje por ventana,
el umbral y la decisión actual (VOZ REAL / VOZ SINTÉTICA).

Fuentes de audio:
  --wav archivo.wav   reproduce el archivo por los parlantes y lo analiza como flujo
  --mic               captura del micrófono en vivo

Incluye una detección de actividad de voz simple por energía: las ventanas por
debajo de --vad-db se marcan como silencio, puntúan 0 y la pantalla muestra
"ESPERANDO VOZ" en lugar de una decisión.

Uso (desde spoof_detection/, con el entorno del proyecto activo):
    python -m scripts.demo --raw-dir $RAW --wav ruta/audio.wav
    python -m scripts.demo --raw-dir $RAW --mic
    python -m scripts.demo --raw-dir $RAW --wav ruta/audio.wav --headless --snapshot results/demo.png

--headless no abre ventana ni espera en tiempo real: procesa todo el archivo y guarda
la figura final (--snapshot), útil para las figuras del informe.

Dependencias extra: matplotlib con interfaz gráfica (`sudo apt install python3-tk`), ffplay
(paquete ffmpeg) para reproducir el archivo, y sounddevice (+ `sudo apt install libportaudio2`)
solo para el micrófono.
El detector se entrena con TODOS los audios de --raw-dir (esto es una demostración, no una
evaluación) y se guarda en models/checkpoints/ para no reentrenar cada vez.
"""

import argparse
import queue
import shutil
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.audio_io import load_audio_16k
from models.detector_a_stream import LFCCSVMDetector, train_detector_a
from scripts.evaluar_otros_tts import parse_raw
from streaming.decision import default_strategies
from streaming.pipeline import StreamingDetector

warnings.filterwarnings("ignore")
SR = 16000

VERDE, ROJO, GRIS = "#2e9e5b", "#d6372f", "#8a8f98"


def cargar_detector(raw_dir, window_sec):
    import joblib
    ckpt = Path("models/checkpoints") / f"detector_a_demo_{window_sec:g}s.joblib"
    if ckpt.exists():
        print(f"Detector A cargado de {ckpt}")
        return LFCCSVMDetector(joblib.load(ckpt))
    entries = [SimpleNamespace(path=p, label="fake" if l == 1 else "real")
               for p, _, l in parse_raw(raw_dir)]
    if not entries:
        sys.exit(f"No hay audios en {raw_dir}")
    print(f"Entrenando Detector A con {len(entries)} audios (una sola vez)...")
    det = train_detector_a(entries, window_sec)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(det.clf, ckpt)
    print(f"Guardado en {ckpt}")
    return det


class Estado:
    """Datos compartidos entre el hilo de procesamiento y la ventana gráfica."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events = []
        self.silent = []
        self.done = False


def construir_stream(det, args, estado):
    hop = args.window_sec * (1 - args.overlap)
    umbral_rms = 10 ** (args.vad_db / 20)

    def score_fn(win, sr):
        silencio = float(np.sqrt(np.mean(win ** 2))) < umbral_rms
        with estado.lock:
            estado.silent.append(silencio)
        return 0.0 if silencio else det.score_window(win, sr)

    decider = default_strategies(args.threshold)[args.strategy]()
    return StreamingDetector(score_fn, SR, args.window_sec, hop, decider)


def procesar(sd, q, estado):
    """Hilo de procesamiento: toma chunks de la cola y los pasa al detector."""
    while True:
        chunk = q.get()
        if chunk is None:
            break
        evs = sd.push(chunk)
        if evs:
            with estado.lock:
                estado.events += evs
    with estado.lock:
        estado.done = True


def fuente_archivo(audio, q, chunk_sec, realtime, play, wav_path=None):
    n = int(chunk_sec * SR)
    if play:
        if shutil.which("ffplay"):
            subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("ffplay no está instalado; se analiza sin sonido (sudo apt install ffmpeg).")
    t0 = time.perf_counter()
    for i in range(0, len(audio), n):
        q.put(audio[i:i + n])
        if realtime:
            wait = t0 + (i + n) / SR - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
    q.put(None)


def fuente_mic(q, device, stop):
    import sounddevice as sd

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SR, channels=1, blocksize=int(0.1 * SR),
                        device=device, callback=cb):
        while not stop.is_set():
            time.sleep(0.1)
    q.put(None)


class Vista:
    def __init__(self, args, titulo):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.args = args
        self.fig, (self.ax, self.ax2) = plt.subplots(
            2, 1, figsize=(11, 6.5), gridspec_kw={"height_ratios": [3, 1.2]})
        self.fig.suptitle(titulo, fontsize=12)
        self.ax.set_ylim(-0.02, 1.02)
        self.ax.set_xlabel("tiempo de audio (s)")
        self.ax.set_ylabel("puntaje P(sintético)")
        self.ax.axhline(args.threshold, color=GRIS, ls="--", lw=1)
        self.ax.text(0.995, args.threshold + 0.02, "umbral", color=GRIS, ha="right",
                     va="bottom", transform=self.ax.get_yaxis_transform(), fontsize=9)
        self.ax.grid(alpha=0.25)
        (self.linea,) = self.ax.plot([], [], color="#444", lw=1, alpha=0.6)
        self.puntos = self.ax.scatter([], [], s=45, zorder=3)
        self.ax2.axis("off")
        self.t_decision = self.ax2.text(0.5, 0.62, "ESCUCHANDO…", ha="center", va="center",
                                        fontsize=34, fontweight="bold", color=GRIS)
        self.t_info = self.ax2.text(0.5, 0.08, "", ha="center", va="center", fontsize=10,
                                    color="#555")
        self.fig.tight_layout(rect=(0, 0, 1, 0.95))

    def dibujar(self, estado):
        with estado.lock:
            evs = list(estado.events)
            sil = list(estado.silent[:len(estado.events)])
        a = self.args
        if not evs:
            return
        x = np.array([e.t_end for e in evs])
        y = np.array([e.score for e in evs])
        cols = [GRIS if s else (ROJO if v >= a.threshold else VERDE) for v, s in zip(y, sil)]
        self.linea.set_data(x, y)
        self.puntos.set_offsets(np.c_[x, y])
        self.puntos.set_color(cols)
        xmax = max(a.span, x[-1] + 1)
        self.ax.set_xlim(xmax - a.span, xmax)

        ult = evs[-1]
        if sil[-1]:
            txt, col = "ESPERANDO VOZ", GRIS
        elif not ult.ready:
            txt, col = "ANALIZANDO…", GRIS
        elif ult.is_fake:
            txt, col = "VOZ SINTÉTICA", ROJO
        else:
            txt, col = "VOZ REAL", VERDE
        self.t_decision.set_text(txt)
        self.t_decision.set_color(col)
        lat = 1000 * np.mean([e.latency_s for e in evs])
        self.t_info.set_text(f"Detector A (LFCC+SVM) · ventana {a.window_sec:g} s · salto "
                             f"{a.window_sec * (1 - a.overlap):g} s · estrategia {a.strategy} · "
                             f"latencia {lat:.1f} ms/ventana")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="audios con los que se entrena el detector")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--wav")
    g.add_argument("--mic", action="store_true")
    ap.add_argument("--device", type=int, default=None, help="índice del micrófono")
    ap.add_argument("--window-sec", type=float, default=1.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--strategy", default="sostenido_k3", choices=list(default_strategies()))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--vad-db", type=float, default=-40.0,
                    help="ventanas con energía (RMS) menor a este nivel en dBFS se marcan como silencio")
    ap.add_argument("--span", type=float, default=20.0, help="segundos visibles en el gráfico")
    ap.add_argument("--no-audio", action="store_true", help="no reproducir el archivo")
    ap.add_argument("--headless", action="store_true", help="sin ventana ni tiempo real")
    ap.add_argument("--snapshot", default=None, help="PNG con la figura final")
    args = ap.parse_args()

    import matplotlib
    if args.headless:
        matplotlib.use("Agg")

    det = cargar_detector(args.raw_dir, args.window_sec)
    estado = Estado()
    sd = construir_stream(det, args, estado)
    q = queue.Queue()
    worker = threading.Thread(target=procesar, args=(sd, q, estado), daemon=True)

    if args.wav:
        audio, _ = load_audio_16k(args.wav)
        titulo = f"Detección en tiempo real — {Path(args.wav).name}"
        if args.headless:
            n = int(0.1 * SR)
            for i in range(0, len(audio), n):
                q.put(audio[i:i + n])
            q.put(None)
            procesar(sd, q, estado)
            args.span = max(args.span, len(audio) / SR + 1)   # figura estática: audio completo
            vista = Vista(args, titulo)
            vista.dibujar(estado)
            if args.snapshot:
                Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)
                vista.fig.savefig(args.snapshot, dpi=160)
                print(f"Figura guardada en {args.snapshot}")
            _resumen(estado, args)
            return
        worker.start()
        threading.Thread(target=fuente_archivo,
                         args=(audio, q, 0.1, True, not args.no_audio, args.wav), daemon=True).start()
    else:
        titulo = "Detección en tiempo real — micrófono"
        stop = threading.Event()
        worker.start()
        threading.Thread(target=fuente_mic, args=(q, args.device, stop), daemon=True).start()

    from matplotlib.animation import FuncAnimation
    if matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps", "cairo", "template"):
        sys.exit("matplotlib no tiene interfaz gráfica en este entorno (backend "
                 f"'{matplotlib.get_backend()}'). Instala tkinter: sudo apt install python3-tk "
                 "y vuelve a abrir la terminal, o usa --headless --snapshot para generar una figura.")
    vista = Vista(args, titulo)
    _anim = FuncAnimation(vista.fig, lambda _f: vista.dibujar(estado), interval=100,
                          cache_frame_data=False)
    try:
        vista.plt.show()
    finally:
        if not args.wav:
            stop.set()
        if args.snapshot:
            vista.dibujar(estado)
            vista.fig.savefig(args.snapshot, dpi=160)
            print(f"Figura guardada en {args.snapshot}")
        _resumen(estado, args)


def _resumen(estado, args):
    with estado.lock:
        evs = list(estado.events)
        sil = list(estado.silent[:len(evs)])
    if not evs:
        print("No se procesó ninguna ventana.")
        return
    n_fake = sum(1 for e, s in zip(evs, sil) if e.ready and e.is_fake)
    prim = next((e for e in evs if e.ready), None)
    print(f"\nVentanas: {len(evs)} | en silencio: {sum(sil)} | decididas 'sintética': {n_fake}")
    if prim:
        print(f"Primera decisión: {'SINTÉTICA' if prim.is_fake else 'REAL'} a los {prim.t_end:.2f} s")
    print(f"Latencia media: {1000 * np.mean([e.latency_s for e in evs]):.1f} ms/ventana")


if __name__ == "__main__":
    main()
