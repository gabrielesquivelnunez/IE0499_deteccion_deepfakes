"""
generar_sinteticos_audio8.py

Genera audios sintéticos con Audio8-TTS (clonación de voz) con los mismos textos que
se usaron con XTTS y Qwen3:

    spkNNN_sintetico_audio8_1.wav   (texto largo)
    spkNNN_sintetico_audio8_2.wav   (transcripción del audio corto del hablante)

Referencia de voz: spkNNN_real_2.wav (audio corto real) y su transcripción
(--textos-cortos-dir/spkNNN.txt). Los hablantes sin alguno de los dos se omiten.

Cada texto se divide en frases de hasta 200 caracteres que se sintetizan por lotes con
audio8_tts_infer.py y luego se unen con una pausa corta. Los audios se escriben en
--out-dir, fuera del conjunto de entrenamiento.

Uso (con venv_audio8 activo, desde spoof_detection/):
    python generacion/generar_sinteticos_audio8.py --raw-dir $RAW --audio8-dir ~/Audio8_TTS \
        --texto-largo textos/largo.txt --textos-cortos-dir textos/cortos \
        --out-dir data/otros_tts/audio8
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

MODELO = "Edge0/Audio8-TTS-Realistic-0.6b"
MAX_CARACTERES = 200
PAUSA_S = 0.25


def leer_texto(path):
    texto = re.sub(r"\s+", " ", Path(path).read_text(encoding="utf-8")).strip()
    if not texto:
        sys.exit(f"El archivo {path} está vacío")
    return texto


def dividir(texto):
    """Separa en frases y agrupa las cortas hasta MAX_CARACTERES."""
    frases = re.split(r"(?<=[.!?;:])\s+", texto)
    bloques, actual = [], ""
    for f in frases:
        if actual and len(actual) + len(f) + 1 > MAX_CARACTERES:
            bloques.append(actual)
            actual = f
        else:
            actual = f"{actual} {f}".strip()
    if actual:
        bloques.append(actual)
    return bloques


def unir(trozos, destino):
    audios, sr = [], None
    for t in trozos:
        a, sr = sf.read(t, dtype="float32")
        audios += [a if a.ndim == 1 else a.mean(axis=1), np.zeros(int(PAUSA_S * sr), dtype=np.float32)]
    sf.write(destino, np.concatenate(audios[:-1]), sr)
    return sum(len(a) for a in audios[:-1]) / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--audio8-dir", required=True, help="carpeta del repositorio Audio8_TTS")
    ap.add_argument("--texto-largo", required=True)
    ap.add_argument("--textos-cortos-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--speakers", nargs="*", default=None, help="por ejemplo: spk005 spk006")
    ap.add_argument("--model", default=MODELO)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    script = Path(args.audio8_dir).expanduser() / "audio8_tts_infer.py"
    if not script.exists():
        sys.exit(f"No se encontró {script}")

    largo = leer_texto(args.texto_largo)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Elementos del lote: un renglón por bloque de texto
    items, destinos = [], {}
    for ref in sorted(Path(args.raw_dir).glob("spk*_real_2.wav")):
        spk = ref.name.split("_")[0]
        if args.speakers and spk not in args.speakers:
            continue
        f_texto = Path(args.textos_cortos_dir) / f"{spk}.txt"
        if not f_texto.exists():
            print(f"{spk}: sin transcripción en {f_texto}, se omite")
            continue
        corto = leer_texto(f_texto)
        for n, texto in ((1, largo), (2, corto)):
            destino = out_dir / f"{spk}_sintetico_audio8_{n}.wav"
            if destino.exists() and not args.overwrite:
                print(f"{destino.name}: ya existe, se omite")
                continue
            ids = []
            for k, bloque in enumerate(dividir(texto)):
                _id = f"{destino.stem}__{k:03d}"
                ids.append(_id)
                items.append({"id": _id, "text": bloque,
                              "reference_audio": str(ref.resolve()), "reference_text": corto})
            destinos[destino] = ids

    if not items:
        print("Nada que generar.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        jsonl = tmp / "prompts.jsonl"
        jsonl.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8")
        partes = tmp / "partes"
        cmd = [sys.executable, str(script), "--model", args.model, "--input-jsonl", str(jsonl),
               "--output-dir", str(partes), "--batch-size", "1", "--dtype", args.dtype,
               "--seed", str(args.seed)]
        print(f"Sintetizando {len(items)} bloques...")
        subprocess.run(cmd, check=False, cwd=script.parent)

        for destino, ids in destinos.items():
            trozos = [partes / f"{i}.wav" for i in ids]
            faltan = [t.name for t in trozos if not t.exists()]
            if faltan:
                print(f"{destino.name}: faltan {len(faltan)} bloque(s), no se genera")
                continue
            print(f"{destino.name}: {unir(trozos, destino):.1f} s")
    print("Listo.")


if __name__ == "__main__":
    main()
