"""
generar_sinteticos_texto.py

Para cada hablante (spkNNN) genera dos audios sintéticos con XTTS v2 que leen
EXACTAMENTE los mismos textos que leyeron los participantes:

    spkNNN_sintetico_xtts_1.wav   (texto largo)
    spkNNN_sintetico_xtts_2.wav   (texto corto)

Así los sintéticos duran lo mismo que los reales (no ~7 s contra 20-70 s) y
real/sintético dicen lo mismo, lo que quita dos confusiones del conjunto actual.
Los sintéticos viejos (frase del Barcelona) no se tocan.

Texto corto: si existe --textos-cortos-dir/spkNNN.txt (lo que dijo esa persona en su
audio corto, ver transcribir_cortos.py) se usa ese; si no, se usa --texto-corto.

Voz de referencia: spkNNN_real_1.wav si existe; si no, el primer audio real del
hablante. Si el archivo de salida ya existe, se salta (use --overwrite para rehacerlo).

Uso (con venv_xtts activo):
    export COQUI_TOS_AGREED=1
    python generar_sinteticos_texto.py --raw-dir RUTA/data/raw \
        --texto-largo textos/largo.txt --texto-corto textos/corto.txt \
        --textos-cortos-dir textos/cortos
"""

import argparse
import re
import sys
from pathlib import Path

LIMITE_FRASE = 230  # XTTS v2 recorta frases largas en español (~239 caracteres)


def leer_texto(path):
    t = Path(path).read_text(encoding="utf-8")
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        sys.exit(f"El archivo {path} está vacío")
    largas = [f for f in re.split(r"(?<=[.!?;:])\s+", t) if len(f) > LIMITE_FRASE]
    if largas:
        print(f"AVISO ({path}): hay {len(largas)} frase(s) de más de {LIMITE_FRASE} caracteres; "
              "XTTS puede recortarlas. Conviene separarlas con puntos.")
    return t


def hablantes(raw_dir):
    """Devuelve {id_hablante: ruta_de_referencia} usando solo audios reales."""
    refs = {}
    for p in sorted(Path(raw_dir).glob("*.wav")):
        if "sintetico" in p.name:
            continue
        spk = p.name.split("_")[0]
        if spk not in refs or p.name.endswith("_real_1.wav"):
            refs[spk] = p
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--texto-largo", required=True)
    ap.add_argument("--texto-corto", required=True)
    ap.add_argument("--textos-cortos-dir", default=None,
                    help="carpeta con spkNNN.txt: lo que dijo cada persona en su audio corto")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    largo, corto = leer_texto(args.texto_largo), leer_texto(args.texto_corto)
    refs = hablantes(args.raw_dir)
    if not refs:
        sys.exit("No se encontraron audios reales en --raw-dir")

    import torch
    from TTS.api import TTS

    orig = torch.load
    torch.load = lambda *a, **k: orig(*a, **{**k, "weights_only": False})

    print("Cargando XTTS v2...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
        "cuda" if torch.cuda.is_available() else "cpu")

    for spk, ref in refs.items():
        corto_spk = corto
        if args.textos_cortos_dir:
            f = Path(args.textos_cortos_dir) / f"{spk}.txt"
            if f.exists():
                corto_spk = leer_texto(f)
        for n, texto in ((1, largo), (2, corto_spk)):
            out = Path(args.raw_dir) / f"{spk}_sintetico_xtts_{n}.wav"
            if out.exists() and not args.overwrite:
                print(f"{out.name}: ya existe, se salta")
                continue
            torch.manual_seed(args.seed + n)
            print(f"{out.name}: referencia {ref.name}")
            try:
                tts.tts_to_file(text=texto, file_path=str(out),
                                speaker_wav=str(ref), language="es")
            except Exception as e:
                print(f"  ERROR en {out.name}: {e}")
    print("Listo.")


if __name__ == "__main__":
    main()
