"""
generar_sinteticos_qwen3.py

Genera audios sintéticos con Qwen3-TTS (modelo Base, clonación de voz) con los mismos
textos que se usaron con XTTS:

    spkNNN_sintetico_qwen3_1.wav   (texto largo)
    spkNNN_sintetico_qwen3_2.wav   (transcripción del audio corto del hablante)

Referencia de voz: spkNNN_real_2.wav (audio corto real) y su transcripción
(--textos-cortos-dir/spkNNN.txt). Los hablantes sin alguno de los dos se omiten.

Los audios se escriben en --out-dir, fuera del conjunto de entrenamiento, para que
no se mezclen con los sintéticos de XTTS en los experimentos.

Uso (con venv_qwen3tts activo, desde spoof_detection/):
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python generacion/generar_sinteticos_qwen3.py --raw-dir $RAW \
        --texto-largo textos/largo.txt --textos-cortos-dir textos/cortos \
        --out-dir data/otros_tts/qwen3
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Decoder

MODELO = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MAX_CARACTERES = 200
PAUSA_S = 0.25

# La decodificación por trozos de 300 pasos no cabe en 4 GB de GPU.
_chunked_decode = Qwen3TTSTokenizerV2Decoder.chunked_decode
Qwen3TTSTokenizerV2Decoder.chunked_decode = (
    lambda self, codes, chunk_size=60, left_context_size=25:
    _chunked_decode(self, codes, chunk_size, left_context_size))


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


def sintetizar(model, prompt, texto, seed):
    torch.manual_seed(seed)
    trozos, sr = [], None
    for bloque in dividir(texto):
        wavs, sr = model.generate_voice_clone(
            text=bloque, language="Spanish", voice_clone_prompt=prompt)
        trozos.append(np.asarray(wavs[0], dtype=np.float32))
        trozos.append(np.zeros(int(PAUSA_S * sr), dtype=np.float32))
    return np.concatenate(trozos[:-1]), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--texto-largo", required=True)
    ap.add_argument("--textos-cortos-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--speakers", nargs="*", default=None, help="por ejemplo: spk005 spk006")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    largo = leer_texto(args.texto_largo)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Qwen3TTSModel.from_pretrained(
        MODELO, device_map=args.device, dtype=getattr(torch, args.dtype))

    for ref in sorted(Path(args.raw_dir).glob("spk*_real_2.wav")):
        spk = ref.name.split("_")[0]
        if args.speakers and spk not in args.speakers:
            continue
        f_texto = Path(args.textos_cortos_dir) / f"{spk}.txt"
        if not f_texto.exists():
            print(f"{spk}: sin transcripción en {f_texto}, se omite")
            continue
        corto = leer_texto(f_texto)
        prompt = model.create_voice_clone_prompt(ref_audio=str(ref), ref_text=corto)

        for n, texto in ((1, largo), (2, corto)):
            out = out_dir / f"{spk}_sintetico_qwen3_{n}.wav"
            if out.exists() and not args.overwrite:
                print(f"{out.name}: ya existe, se omite")
                continue
            try:
                audio, sr = sintetizar(model, prompt, texto, args.seed + n)
            except Exception as e:
                print(f"{out.name}: error ({e})")
                torch.cuda.empty_cache()
                continue
            sf.write(out, audio, sr)
            print(f"{out.name}: {len(audio) / sr:.1f} s")
    print("Listo.")


if __name__ == "__main__":
    main()
