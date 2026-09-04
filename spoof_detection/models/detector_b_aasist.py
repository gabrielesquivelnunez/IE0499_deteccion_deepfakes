"""
detector_b_aasist.py

Detector B del proyecto: AASIST preentrenado, usado por transferencia
(transfer learning) sobre el conjunto de voces del laboratorio.

Requisitos:
    1. Ejecutar primero `scripts/setup_aasist.sh` para clonar el repo
       oficial (clovaai/aasist) dentro de external/aasist/. Ese repo ya
       incluye los pesos preentrenados sobre ASVspoof2019 LA en
       external/aasist/models/weights/{AASIST.pth, AASIST-L.pth}.
    2. Instalar PyTorch (no se incluye en requirements.txt por defecto
       porque es pesado; instalar según tu hardware, ver
       https://pytorch.org/get-started/locally/).

Este script NO reentrena la arquitectura desde cero: carga los pesos
oficiales y expone dos modos de uso:
    - Extractor congelado + puntaje directo (evaluación "zero-shot").
    - Fine-tuning: descongela las últimas capas y sigue entrenando con
      los datos del laboratorio (recomendado una vez que haya conjunto
      de datos propio disponible).

IMPORTANTE: el modelo AASIST oficial espera audio crudo a 16 kHz, de
longitud fija (nb_samp = 64600 muestras ≈ 4.04 s, ver
external/aasist/config/AASIST.conf). Nuestras ventanas de duración
variable (0.5, 1, 2, 3, 5 s) se recortan/rellenan a esa longitud antes de
pasar por el modelo. Esta restricción de longitud fija es justamente uno
de los puntos que hay que documentar como limitación al comparar AASIST
contra el Enfoque A (LFCC+SVM), que sí acepta ventanas de longitud libre.
"""

import json
import sys
from pathlib import Path

import numpy as np

AASIST_REPO = Path(__file__).resolve().parents[1] / "external" / "aasist"
sys.path.insert(0, str(AASIST_REPO))


def _lazy_import_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "PyTorch no está instalado. Instalarlo con:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "(o la variante con CUDA correspondiente a tu GPU)."
        ) from e


class AASISTDetector:
    """Wrapper alrededor del modelo oficial AASIST para usarlo por
    transferencia en este proyecto."""

    def __init__(self, config_name: str = "AASIST.conf", device: str | None = None):
        torch = _lazy_import_torch()
        from models.AASIST import Model  # noqa: E402  (import del repo externo)

        config_path = AASIST_REPO / "config" / config_name
        with open(config_path) as f:
            config = json.load(f)

        self.model_config = config["model_config"]
        self.nb_samp = self.model_config["nb_samp"]
        self.sr = 16000  # AASIST fue entrenado a 16 kHz

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(self.model_config).to(self.device)

        weights_path = AASIST_REPO / "models" / "weights" / config["model_path"].split("/")[-1]
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"AASIST cargado desde {weights_path.name} en {self.device}")

    def _fit_length(self, audio: np.ndarray) -> np.ndarray:
        """Recorta o repite (padding circular, como hace el repo oficial)
        el audio a la longitud fija que espera el modelo."""
        if len(audio) >= self.nb_samp:
            return audio[: self.nb_samp]
        n_repeats = int(np.ceil(self.nb_samp / len(audio)))
        return np.tile(audio, n_repeats)[: self.nb_samp]

    def score_window(self, audio: np.ndarray, sr: int) -> float:
        """Devuelve la probabilidad de que la ventana sea SINTÉTICA (spoof).
        Asume audio mono. Si sr != 16000, hay que remuestrear antes de
        llamar a esta función (ver features/windowing.py + resample)."""
        torch = _lazy_import_torch()
        if sr != self.sr:
            raise ValueError(f"AASIST espera audio a {self.sr} Hz, se recibió {sr} Hz. "
                              f"Remuestrear antes de llamar a score_window().")

        audio_fixed = self._fit_length(audio.astype(np.float32))
        x = torch.from_numpy(audio_fixed).unsqueeze(0).to(self.device)  # (1, nb_samp)

        with torch.no_grad():
            _, logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            spoof_prob = probs[0, 1].item()  # índice 1 = clase "spoof" en el repo oficial
        return spoof_prob

    def unfreeze_for_finetuning(self, n_last_layers: int = 2):
        """Descongela solo las últimas capas (out_layer y las capas de
        atención más cercanas a la salida) para hacer fine-tuning barato,
        dejando el resto del encoder congelado. Ajustar según los
        resultados iniciales del entregable 2."""
        for p in self.model.parameters():
            p.requires_grad = False

        trainable = [self.model.out_layer]
        if n_last_layers >= 2:
            trainable += [self.model.HtrgGAT_layer_ST12, self.model.HtrgGAT_layer_ST22]

        for layer in trainable:
            for p in layer.parameters():
                p.requires_grad = True

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Fine-tuning habilitado en {len(trainable)} capas ({n_trainable:,} parámetros entrenables)")


if __name__ == "__main__":
    # Prueba rápida: cargar el modelo y correr un audio de ruido aleatorio,
    # solo para confirmar que la carga de pesos y el forward funcionan.
    detector = AASISTDetector()
    dummy_audio = np.random.randn(16000 * 2).astype(np.float32) * 0.01  # 2s de "ruido"
    score = detector.score_window(dummy_audio, sr=16000)
    print(f"Puntaje de prueba (esperado: cercano a 0.5, es audio aleatorio): {score:.3f}")
