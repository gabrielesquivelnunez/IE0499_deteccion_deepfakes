"""
transcribir_cortos.py

Transcribe los audios cortos (spkNNN_real_2.wav) de los participantes para que
generar_sinteticos_texto.py haga que el sintético diga lo mismo que dijo cada
persona. Escribe una transcripción por hablante en --out-dir/spkNNN.txt.

Uso (cualquier venv, por ejemplo venv_xtts):
    pip install faster-whisper
    python transcribir_cortos.py --raw-dir $RAW --out-dir textos/cortos

REVISA los .txt a mano (los nombres propios y las palabras raras se transcriben
mal) y corrige lo que haga falta antes de generar los sintéticos.
Estos textos son lo que dijo cada persona: no los subas a GitHub.
"""

import argparse
import sys
from pathlib import Path

# Se corre desde spoof_detection/: se usa el mismo cargador (16 kHz mono) que el proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features.audio_io import load_audio_16k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="small")
    args = ap.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("Falta faster-whisper:  pip install faster-whisper")

    files = sorted(Path(args.raw_dir).glob("spk*_real_2.wav"))
    if not files:
        sys.exit("No se encontraron archivos spk*_real_2.wav en --raw-dir")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    for f in files:
        audio, _ = load_audio_16k(f)  # arreglo float32 a 16 kHz: evita decodificar con av
        segs, _ = model.transcribe(audio, language="es", beam_size=5)
        texto = " ".join(s.text.strip() for s in segs).strip()
        out = Path(args.out_dir) / f"{f.name.split('_')[0]}.txt"
        out.write_text(texto + "\n", encoding="utf-8")
        print(f"{f.name}: {len(texto)} caracteres -> {out}")
        print(f"   {texto[:120]}{'...' if len(texto) > 120 else ''}")


if __name__ == "__main__":
    main()
