"""
run_smoke_test.py

Prueba rápida de humo (smoke test): corre TODO el pipeline del Detector A
sobre audio sintético de juguete (no es habla real, son tonos generados),
solo para confirmar que la instalación y el código funcionan de punta a
punta ANTES de conectar el dataset real del laboratorio.

Uso:
    python examples/run_smoke_test.py

Qué hace:
    1. Genera un puñado de audios .wav sintéticos en examples/dummy_data/raw
       (si no existen ya).
    2. Corre data/prepare_dataset.py para armar las particiones.
    3. Entrena y evalúa el Detector A (LFCC + SVM) sobre esos datos de juguete.

Nota: los resultados de este smoke test NO significan nada sobre el
desempeño real del detector (los datos son artificiales y trivialmente
separables a propósito). Solo sirve para validar que el código corre.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "examples" / "dummy_data" / "raw"


def make_tone(freq, dur_sec, sr, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, dur_sec, int(sr * dur_sec), endpoint=False)
    sig = 0.3 * np.sin(2 * np.pi * freq * t) + 0.15 * np.sin(2 * np.pi * 2 * freq * t)
    sig += noise * rng.standard_normal(len(t))
    return sig.astype(np.float32)


def generate_dummy_data(sr=16000, n_speakers=4, n_utts=3):
    (RAW_DIR / "real").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "fake").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    for i in range(n_speakers):
        spk = f"speaker{i+1:02d}"
        for j in range(n_utts):
            dur = rng.uniform(3, 6)
            real_sig = make_tone(110 + i * 10, dur, sr, noise=0.03, seed=i * 10 + j)
            sf.write(RAW_DIR / "real" / f"{spk}_utt{j:03d}.wav", real_sig, sr)

            fake_sig = make_tone(110 + i * 10, dur, sr, noise=0.005, seed=i * 10 + j + 100)
            fake_sig += 0.03 * np.sin(2 * np.pi * 4200 * np.linspace(0, dur, len(fake_sig)))
            sf.write(RAW_DIR / "fake" / f"tts_{spk}_utt{j:03d}.wav", fake_sig, sr)

    print(f"Datos de prueba generados en {RAW_DIR}")


def main():
    if not any(RAW_DIR.glob("**/*.wav")):
        generate_dummy_data()
    else:
        print(f"Ya existen audios en {RAW_DIR}, se reutilizan.")

    print("\n=== Paso 1: organizar particiones ===")
    subprocess.run(
        [sys.executable, "data/prepare_dataset.py"],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT)},
        check=True,
    ) if False else None
    # se corre con --raw-dir apuntando a los datos de juguete
    sys.path.insert(0, str(ROOT))
    from data.prepare_dataset import main as prepare_main
    prepare_main(raw_dir=str(RAW_DIR), out_dir=str(ROOT / "examples" / "dummy_data" / "splits"))

    print("\n=== Paso 2: entrenar y evaluar Detector A (LFCC + SVM) ===")
    subprocess.run(
        [
            sys.executable, "-m", "models.detector_a_lfcc_svm",
            "--train", "examples/dummy_data/splits/train.csv",
            "--val", "examples/dummy_data/splits/val.csv",
            "--test", "examples/dummy_data/splits/test.csv",
            "--window-sec", "1.0", "--hop-sec", "1.0",
            "--model-out", "examples/dummy_data/detector_a_smoke_test.joblib",
        ],
        cwd=ROOT,
        check=True,
    )

    print("\nSmoke test completo. Si llegaste hasta aquí sin errores, "
          "el entorno está listo para conectar el dataset real.")


if __name__ == "__main__":
    main()
