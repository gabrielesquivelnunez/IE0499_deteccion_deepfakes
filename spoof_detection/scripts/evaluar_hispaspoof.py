"""
evaluar_hispaspoof.py

Evaluación SIN ENTRENAR del Detector A (LFCC + SVM) con la partición de detección de
HISPASpoof (voces reales de varios acentos del español y sintéticos de seis generadores).
El detector es el mismo del demostrador: entrenado con los 57 audios del proyecto
(reales + XTTS v2). Ningún audio de HISPASpoof interviene en el entrenamiento.

Por generador reporta, con ventanas de --window-sec (50 % de solape):
  - EER por ventana contra las ventanas de las voces reales (natural).
  - % de ventanas con puntaje sobre el umbral.
  - % de archivos marcados como sintéticos (puntaje medio del archivo >= umbral).
Para las voces reales ("natural") ese último porcentaje es la tasa de falsas alarmas.

Son frases sueltas de pocos segundos, no flujos largos, por lo que la decisión es por archivo.

Preparación (una vez):
    from huggingface_hub import hf_hub_download
    hf_hub_download("purdueviperlab/HISPASpoof", "detection/val.tar", repo_type="dataset")
    hf_hub_download("purdueviperlab/HISPASpoof", "detection/protocols/val_metadata.csv", repo_type="dataset")
Los comandos imprimen la ruta local de cada archivo.

Uso (desde spoof_detection/, con el entorno del proyecto activo):
    python -m scripts.evaluar_hispaspoof --raw-dir $RAW --tar RUTA/val.tar \
        --metadata RUTA/val_metadata.csv [--max-por-generador 200]
"""

import argparse
import csv
import io
import random
import re
import sys
import tarfile
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.audio_io import TARGET_SR, to_mono_16k
from features.windowing import make_windows
from metrics.metrics import compute_eer
from scripts.demo import cargar_detector

warnings.filterwarnings("ignore")

NAME_RE = re.compile(r"^detection_(?P<gen>.+?)_speaker_(?P<spk>\d+)_sentence_(?P<sent>\d+)\.wav$")


def leer_metadata(path):
    """Devuelve {archivo: (generador, etiqueta)}; etiqueta 0 = real, 1 = sintético."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = NAME_RE.match(r["filename"])
            if not m:
                continue
            out[r["filename"]] = (m.group("gen"), int(r["label"]))
    return out


def elegir(meta, max_por_gen, seed):
    por_gen = defaultdict(list)
    for fn, (g, _) in meta.items():
        por_gen[g].append(fn)
    rng = random.Random(seed)
    elegidos = set()
    for g, fns in por_gen.items():
        fns = sorted(fns)
        if max_por_gen and len(fns) > max_por_gen:
            fns = rng.sample(fns, max_por_gen)
        elegidos.update(fns)
    return elegidos


def fmt(x, nd=1):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="audios con los que se entrena el detector")
    ap.add_argument("--tar", required=True, help="detection/val.tar (o test.tar, train.tar)")
    ap.add_argument("--metadata", required=True, help="CSV de protocolo correspondiente")
    ap.add_argument("--window-sec", type=float, default=1.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-por-generador", type=int, default=0,
                    help="máximo de archivos por generador (0 = todos)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    meta = leer_metadata(args.metadata)
    if not meta:
        sys.exit("El CSV no tiene archivos con el formato esperado (detection_<gen>_speaker_N_sentence_M.wav)")
    elegidos = elegir(meta, args.max_por_generador, args.seed)
    print(f"{len(elegidos)} de {len(meta)} archivos seleccionados")

    det = cargar_detector(args.raw_dir, args.window_sec)
    hop = args.window_sec * (1 - args.overlap)

    puntajes = defaultdict(list)       # generador -> lista de arrays (uno por archivo)
    sin_ventanas = 0
    hechos = 0
    with tarfile.open(args.tar) as tf:
        for m in tf:
            base = Path(m.name).name
            if not m.isfile() or base not in elegidos:
                continue
            try:
                audio, sr = sf.read(io.BytesIO(tf.extractfile(m).read()))
            except Exception as e:
                print(f"  no se pudo leer {base}: {e}")
                continue
            audio = to_mono_16k(audio, sr)
            wins = make_windows(audio, TARGET_SR, args.window_sec, hop)
            if not wins:
                sin_ventanas += 1
                continue
            sc = np.array([det.score_window(w, TARGET_SR) for w in wins])
            puntajes[meta[base][0]].append(sc)
            hechos += 1
            if hechos % 200 == 0:
                print(f"  {hechos} archivos procesados")
    print(f"Procesados: {hechos} | sin ventanas (muy cortos): {sin_ventanas}")

    if "natural" not in puntajes:
        sys.exit("No hay voces reales ('natural') en la muestra; no se pueden calcular las métricas.")

    tau = args.threshold
    real_win = np.concatenate(puntajes["natural"])
    fa = float(np.mean([s.mean() >= tau for s in puntajes["natural"]]))

    print(f"\nDetector A entrenado con el proyecto | ventana {args.window_sec:g} s, "
          f"solape {args.overlap:.0%}, umbral {tau}")
    print(f"{'generador':12s} {'archivos':>8s} {'EER_vent%':>9s} {'vent>umbral%':>12s} {'archivos_falsos%':>16s}")
    rows = []
    orden = ["natural"] + sorted(g for g in puntajes if g != "natural")
    for g in orden:
        arrs = puntajes[g]
        win = np.concatenate(arrs)
        over = float((win >= tau).mean())
        flagged = float(np.mean([s.mean() >= tau for s in arrs]))
        if g == "natural":
            eer = float("nan")
        else:
            eer, _ = compute_eer(real_win, win)
        etiqueta = g + (" (FA)" if g == "natural" else "")
        print(f"{etiqueta:12s} {len(arrs):8d} {fmt(eer * 100):>9s} {fmt(over * 100):>12s} {fmt(flagged * 100):>16s}")
        rows.append({"generador": g, "archivos": len(arrs), "ventanas": len(win),
                     "eer_ventana": None if eer != eer else round(eer, 4),
                     "frac_ventanas_sobre_umbral": round(over, 4),
                     "frac_archivos_marcados_sinteticos": round(flagged, 4),
                     "window_sec": args.window_sec, "overlap": args.overlap, "threshold": tau})
    print(f"\nFalsas alarmas en voces reales: {fa * 100:.1f} % de {len(puntajes['natural'])} archivos")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"hispaspoof_A_{Path(args.tar).stem}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Resultados guardados en {out}")


if __name__ == "__main__":
    main()
