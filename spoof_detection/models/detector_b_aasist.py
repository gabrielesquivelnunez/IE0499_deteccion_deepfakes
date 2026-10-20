"""
detector_b_aasist.py

Detector B: AASIST / AASIST-L (clovaai/aasist) como extractor+clasificador.

CORRECCIÓN (oct 2026): en el repo oficial la etiqueta 1 es BONAFIDE y 0 es
SPOOF (data_utils.py: `1 if label == "bonafide" else 0`; main.py puntúa con
`batch_out[:, 1]` = puntaje de bonafide). Antes se tomaba el índice 1 como
"spoof", así que los puntajes de B salían INVERTIDOS. Ahora
score_window() devuelve P(spoof) = softmax(logits)[0].

Requisitos: external/aasist/ clonado (scripts/setup_aasist.sh) y PyTorch.
El modelo espera audio crudo a 16 kHz y longitud fija nb_samp (64600 ≈ 4.04 s):
las ventanas se recortan o repiten (tile) a esa longitud.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

AASIST_REPO = Path(__file__).resolve().parents[1] / "external" / "aasist"

SPOOF_INDEX = 0      # índice de la clase "spoof" en la salida de AASIST
BONAFIDE_INDEX = 1   # índice de la clase "bonafide"


def _lazy_import_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("PyTorch no está instalado (pip install torch).") from e


def _load_official_aasist_model_class():
    """Carga models/AASIST.py del repo oficial POR RUTA, para no chocar con
    nuestro paquete local `models`."""
    path = AASIST_REPO / "models" / "AASIST.py"
    spec = importlib.util.spec_from_file_location("aasist_official_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


class AASISTDetector:
    def __init__(self, config_name: str = "AASIST-L.conf", device: str | None = None,
                 verbose: bool = True):
        torch = _lazy_import_torch()
        Model = _load_official_aasist_model_class()

        with open(AASIST_REPO / "config" / config_name) as f:
            config = json.load(f)

        self.model_config = config["model_config"]
        self.nb_samp = self.model_config["nb_samp"]
        self.sr = 16000
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(self.model_config).to(self.device)

        self.weights_path = AASIST_REPO / "models" / "weights" / config["model_path"].split("/")[-1]
        self.reset_weights()
        if verbose:
            print(f"AASIST cargado desde {self.weights_path.name} en {self.device}")

    def reset_weights(self):
        """Vuelve a los pesos preentrenados (útil entre folds de fine-tuning)."""
        torch = _lazy_import_torch()
        state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        for p in self.model.parameters():
            p.requires_grad = True
        self.model.eval()

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def _fit_length(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) >= self.nb_samp:
            return audio[: self.nb_samp]
        n_repeats = int(np.ceil(self.nb_samp / len(audio)))
        return np.tile(audio, n_repeats)[: self.nb_samp]

    def score_window(self, audio: np.ndarray, sr: int) -> float:
        """P(sintético) de una ventana mono a 16 kHz."""
        torch = _lazy_import_torch()
        if sr != self.sr:
            raise ValueError(f"AASIST espera {self.sr} Hz, se recibió {sr} Hz.")
        x = torch.from_numpy(self._fit_length(audio.astype(np.float32))).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, logits = self.model(x)
            return torch.softmax(logits, dim=-1)[0, SPOOF_INDEX].item()

    def unfreeze_for_finetuning(self, n_last_layers: int = 2):
        """Congela todo salvo la capa de salida (y, con n_last_layers>=2, las dos
        últimas capas de atención heterogénea)."""
        for p in self.model.parameters():
            p.requires_grad = False
        trainable = [self.model.out_layer]
        if n_last_layers >= 2:
            trainable += [self.model.HtrgGAT_layer_ST12, self.model.HtrgGAT_layer_ST22]
        for layer in trainable:
            for p in layer.parameters():
                p.requires_grad = True
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Fine-tuning en {len(trainable)} capas ({n:,} parámetros entrenables)")


if __name__ == "__main__":
    det = AASISTDetector()
    print("P(spoof) de ruido:", round(det.score_window(np.random.randn(32000).astype(np.float32) * 0.01, 16000), 3))
