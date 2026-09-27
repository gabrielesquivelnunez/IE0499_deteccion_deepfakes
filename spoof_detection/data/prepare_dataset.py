"""
prepare_dataset.py

Organiza el conjunto de voces naturales y sintéticas en particiones,
EVITANDO que el mismo locutor aparezca en más de una partición. Esto es
clave: si un locutor aparece tanto en train como en test, el detector
puede "memorizar" rasgos de ese locutor en vez de aprender a distinguir
real/sintético en general, y los resultados de evaluación quedarían
artificialmente optimistas.

Soporta DOS formatos de entrada:

1. Carpeta plana (convención real del laboratorio, voice-similarity-eval):

    data/raw/
    ├── daniel_ref1.wav          -> locutor=daniel, real
    ├── daniel_sintetico_xtts.wav -> locutor=daniel, fake
    ├── nayelhi_real_1.wav       -> locutor=nayelhi, real
    └── nayelhi_sintetico_xtts.wav -> locutor=nayelhi, fake

   El locutor es el texto antes del primer "_"; es "fake" si el nombre
   contiene la palabra "sintetico", y "real" en cualquier otro caso.

2. Carpetas separadas real/fake (útil si más adelante se complementa con
   otro corpus, p. ej. ASVspoof, organizado así):

    data/raw/
    ├── real/...
    └── fake/...

Y DOS estrategias de partición:

- split_dataset(): train/val/test agrupado por locutor (GroupShuffleSplit).
  Requiere varios locutores por partición para ser representativo —
  con pocos locutores (menos de ~8-10) los porcentajes dejan de tener
  sentido real (puede tocarte una sola partición con un único locutor).

- leave_one_speaker_out_splits(): evaluación leave-one-speaker-out (LOSO):
  por cada locutor, se entrena con TODOS los demás y se evalúa solo en
  ese locutor, rotando. Es la alternativa correcta mientras el dataset
  tenga pocos locutores (como el actual: 4), porque aprovecha todos los
  datos disponibles sin inventar una partición arbitraria pequeña.
"""

import csv
import re
from pathlib import Path
from dataclasses import dataclass

from sklearn.model_selection import GroupShuffleSplit


@dataclass
class AudioEntry:
    path: str
    label: str       # "real" o "fake"
    group_id: str     # locutor, para evitar fuga de información


def parse_speaker_and_label(filepath: Path) -> tuple[str, str]:
    """
    Convención real del laboratorio (voice-similarity-eval): el locutor es
    el texto antes del primer "_", y el archivo es "fake" si el nombre
    contiene "sintetico" (p. ej. "..._sintetico_xtts.wav").

    Ejemplos:
        daniel_ref1.wav            -> ("daniel", "real")
        daniel_sintetico_xtts.wav  -> ("daniel", "fake")
        nayelhi_real_1.wav         -> ("nayelhi", "real")
    """
    stem = filepath.stem
    match = re.match(r"^([a-zA-Z]+)_", stem)
    speaker = match.group(1).lower() if match else stem
    label = "fake" if "sintetico" in stem.lower() else "real"
    return speaker, label


def scan_dataset(raw_dir: str) -> list[AudioEntry]:
    """
    Detecta automáticamente el formato de la carpeta:
    - Si existen raw_dir/real/ y/o raw_dir/fake/, los usa (carpetas separadas).
    - Si no, trata raw_dir como carpeta plana y parsea cada nombre de
      archivo con parse_speaker_and_label().
    """
    raw_path = Path(raw_dir)
    entries = []

    has_subfolders = (raw_path / "real").exists() or (raw_path / "fake").exists()

    if has_subfolders:
        for label in ("real", "fake"):
            folder = raw_path / label
            if not folder.exists():
                continue
            for wav_path in sorted(folder.glob("*.wav")):
                speaker, _ = parse_speaker_and_label(wav_path)
                entries.append(AudioEntry(path=str(wav_path), label=label, group_id=speaker))
    else:
        for wav_path in sorted(raw_path.glob("*.wav")):
            speaker, label = parse_speaker_and_label(wav_path)
            entries.append(AudioEntry(path=str(wav_path), label=label, group_id=speaker))

    return entries


def split_dataset(entries: list[AudioEntry], test_size: float = 0.2,
                   val_size: float = 0.1, random_state: int = 42):
    """
    Divide en train/val/test agrupando por locutor. Requiere suficientes
    locutores distintos (idealmente 8-10+) para que los porcentajes
    tengan sentido real; con pocos, usar leave_one_speaker_out_splits().
    """
    n_groups = len({e.group_id for e in entries})
    if n_groups < 6:
        print(f"AVISO: solo hay {n_groups} locutores distintos. Un split "
              f"train/val/test con tan pocos grupos no es representativo "
              f"(alguna partición puede quedar con un solo locutor o vacía). "
              f"Considerar usar leave_one_speaker_out_splits() en su lugar.")

    groups = [e.group_id for e in entries]

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss_test.split(entries, groups=groups))

    trainval_entries = [entries[i] for i in trainval_idx]
    trainval_groups = [groups[i] for i in trainval_idx]

    relative_val_size = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(trainval_entries, groups=trainval_groups))

    train = [trainval_entries[i] for i in train_idx]
    val = [trainval_entries[i] for i in val_idx]
    test = [entries[i] for i in test_idx]

    return train, val, test


def leave_one_speaker_out_splits(entries: list[AudioEntry]):
    """
    Generador de particiones leave-one-speaker-out (LOSO): para cada
    locutor presente, produce (locutor_excluido, train, test), donde
    train son TODOS los demás locutores y test es únicamente ese locutor.

    Es la estrategia recomendada mientras el dataset tenga pocos
    locutores (actualmente 4): en vez de "desperdiciar" datos en una
    partición de test fija y chiquita, se evalúa sobre cada locutor por
    turnos y luego se promedian las métricas (EER, F1, etc.) de las N
    rondas. No hay partición de validación separada en este modo —usar
    los mismos datos de train para ajustar hiperparámetros con cautela,
    o simplemente reportar el promedio de las N rondas como resultado
    principal de esta etapa temprana del proyecto.
    """
    speakers = sorted({e.group_id for e in entries})
    for held_out in speakers:
        train = [e for e in entries if e.group_id != held_out]
        test = [e for e in entries if e.group_id == held_out]
        yield held_out, train, test


def _assert_no_leakage(train, val, test):
    g_train = {e.group_id for e in train}
    g_val = {e.group_id for e in val}
    g_test = {e.group_id for e in test}
    assert not (g_train & g_val), f"Fuga train/val: {g_train & g_val}"
    assert not (g_train & g_test), f"Fuga train/test: {g_train & g_test}"
    assert not (g_val & g_test), f"Fuga val/test: {g_val & g_test}"


def save_split_csv(entries: list[AudioEntry], out_path: str):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "group_id"])
        for e in entries:
            writer.writerow([e.path, e.label, e.group_id])


def main(raw_dir: str = "data/raw", out_dir: str = "data/splits", force_loso: bool = False):
    entries = scan_dataset(raw_dir)
    if not entries:
        print(f"No se encontraron archivos .wav en {raw_dir}. Verificar la ruta.")
        return

    n_speakers = len({e.group_id for e in entries})
    n_real = sum(1 for e in entries if e.label == "real")
    n_fake = sum(1 for e in entries if e.label == "fake")
    print(f"Total: {len(entries)} archivos | {n_real} reales, {n_fake} sintéticos | {n_speakers} locutores")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if force_loso or n_speakers < 6:
        print(f"\nUsando leave-one-speaker-out ({n_speakers} locutores encontrados).")
        for held_out, train, test in leave_one_speaker_out_splits(entries):
            save_split_csv(train, f"{out_dir}/loso_{held_out}_train.csv")
            save_split_csv(test, f"{out_dir}/loso_{held_out}_test.csv")
            print(f"  ronda '{held_out}': train={len(train)} archivos, test={len(test)} archivos")
        print(f"\nParticiones LOSO guardadas en {out_dir}/ (una por locutor excluido).")
    else:
        train, val, test = split_dataset(entries)
        _assert_no_leakage(train, val, test)
        save_split_csv(train, f"{out_dir}/train.csv")
        save_split_csv(val, f"{out_dir}/val.csv")
        save_split_csv(test, f"{out_dir}/test.csv")
        print(f"  train: {len(train)}  ({len({e.group_id for e in train})} locutores)")
        print(f"  val:   {len(val)}  ({len({e.group_id for e in val})} locutores)")
        print(f"  test:  {len(test)}  ({len({e.group_id for e in test})} locutores)")
        print(f"Particiones guardadas en {out_dir}/")


if __name__ == "__main__":
    main()