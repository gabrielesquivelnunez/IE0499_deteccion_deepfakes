"""
prepare_dataset.py

Organiza el conjunto de voces naturales y sintéticas en particiones de
entrenamiento, validación y prueba, EVITANDO que el mismo locutor (o el
mismo sistema TTS) aparezca en más de una partición.

Formato esperado de entrada:
    data/raw/
    ├── real/
    │   ├── speaker01_utt001.wav
    │   ├── speaker01_utt002.wav
    │   └── speaker02_utt001.wav
    └── fake/
        ├── ttsA_speaker01_utt001.wav
        └── ttsB_speaker02_utt001.wav

El "group_id" (locutor o sistema) se extrae del nombre de archivo con una
función configurable — ajustar `default_group_extractor` a la convención
de nombres real del dataset del laboratorio.
"""

import csv
import re
from pathlib import Path
from dataclasses import dataclass

from sklearn.model_selection import GroupShuffleSplit


@dataclass
class AudioEntry:
    path: str
    label: str      # "real" o "fake"
    group_id: str    # locutor o sistema TTS, para evitar fuga de información


def default_group_extractor(filepath: Path) -> str:
    """
    Extrae un identificador de grupo (locutor) del nombre del archivo.
    Ajustar esta función a la convención real de nombres del dataset.
    """
    stem = filepath.stem
    match = re.search(r"(speaker\d+)", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # Si no matchea el patrón esperado, se usa el nombre completo como
    # grupo propio (queda aislado en una sola partición).
    return stem


def scan_dataset(raw_dir: str, group_extractor=default_group_extractor) -> list[AudioEntry]:
    raw_path = Path(raw_dir)
    entries = []
    for label, subdir in [("real", "real"), ("fake", "fake")]:
        folder = raw_path / subdir
        if not folder.exists():
            continue
        for wav_path in sorted(folder.glob("*.wav")):
            entries.append(AudioEntry(
                path=str(wav_path),
                label=label,
                group_id=group_extractor(wav_path),
            ))
    return entries


def split_dataset(entries: list[AudioEntry], test_size: float = 0.2,
                   val_size: float = 0.1, random_state: int = 42):
    """
    Divide en train/val/test agrupando por group_id, usa
    GroupShuffleSplit para que ningún grupo
    quede repartido entre particiones distintas.
    """
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


def main(raw_dir: str = "data/raw", out_dir: str = "data/splits"):
    entries = scan_dataset(raw_dir)
    if not entries:
        print(f"No se encontraron archivos .wav en {raw_dir}/real y {raw_dir}/fake. "
              f"Verificar la estructura de carpetas.")
        return

    train, val, test = split_dataset(entries)
    _assert_no_leakage(train, val, test)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_split_csv(train, f"{out_dir}/train.csv")
    save_split_csv(val, f"{out_dir}/val.csv")
    save_split_csv(test, f"{out_dir}/test.csv")

    print(f"Total de archivos: {len(entries)}")
    print(f"  train: {len(train)}  ({len({e.group_id for e in train})} grupos)")
    print(f"  val:   {len(val)}  ({len({e.group_id for e in val})} grupos)")
    print(f"  test:  {len(test)}  ({len({e.group_id for e in test})} grupos)")
    print(f"Particiones guardadas en {out_dir}/")


if __name__ == "__main__":
    main()
