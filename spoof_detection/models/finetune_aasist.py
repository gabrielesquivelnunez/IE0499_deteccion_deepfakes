"""
finetune_aasist.py

Fine-tuning ligero de AASIST-L: se congela casi todo el modelo y solo se
entrenan la capa de salida y las dos últimas capas de atención (ver
AASISTDetector.unfreeze_for_finetuning). El modelo se mantiene en modo eval()
(BatchNorm con estadísticas preentrenadas, sin dropout) porque el conjunto es
diminuto; solo se actualizan los parámetros descongelados.

Etiquetas del modelo: 0 = spoof, 1 = bonafide (convención del repo oficial).
"""

import numpy as np

from features.audio_io import load_audio_16k
from features.windowing import make_windows
from models.detector_b_aasist import SPOOF_INDEX, BONAFIDE_INDEX


def make_training_windows(entries, window_sec: float, hop_sec: float | None = None):
    """Ventanas (con solape, para tener más datos) de los archivos de
    entrenamiento. Devuelve lista de (ventana, es_sintético)."""
    hop_sec = hop_sec or window_sec / 2
    out = []
    for e in entries:
        audio, sr = load_audio_16k(e.path)
        for w in make_windows(audio, sr, window_sec, hop_sec):
            out.append((w, int(e.label == "fake")))
    return out


def finetune(detector, windows, epochs: int = 8, lr: float = 1e-4,
             batch_size: int = 16, micro_batch: int = 4, seed: int = 0, verbose: bool = True):
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    detector.unfreeze_for_finetuning()
    params = [p for p in detector.model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    labels = np.array([y for _, y in windows])
    n_fake, n_real = labels.sum(), (1 - labels).sum()
    # pesos inversos a la frecuencia, en el orden de índices de AASIST
    weight = torch.zeros(2)
    weight[SPOOF_INDEX] = len(labels) / (2 * max(n_fake, 1))
    weight[BONAFIDE_INDEX] = len(labels) / (2 * max(n_real, 1))
    weight = weight.to(detector.device)

    detector.model.eval()
    for ep in range(epochs):
        order = rng.permutation(len(windows))
        total = 0.0
        for i in range(0, len(order), batch_size):
            batch = [windows[j] for j in order[i:i + batch_size]]
            x = np.stack([detector._fit_length(w.astype(np.float32) * 10 ** (rng.uniform(-6, 6) / 20))
                          for w, _ in batch])  # ganancia aleatoria ±6 dB
            # etiqueta de AASIST: sintético -> SPOOF_INDEX, real -> BONAFIDE_INDEX
            y = np.array([SPOOF_INDEX if lab else BONAFIDE_INDEX for _, lab in batch])
            opt.zero_grad()
            batch_loss = 0.0
            # micro-lotes con acumulación de gradiente: mismo lote efectivo,
            # pero la GPU (4 GB) no se queda sin memoria
            for k in range(0, len(batch), micro_batch):
                xt = torch.from_numpy(x[k:k + micro_batch]).to(detector.device)
                yt = torch.from_numpy(y[k:k + micro_batch]).long().to(detector.device)
                _, logits = detector.model(xt)
                loss = F.cross_entropy(logits, yt, weight=weight) * (len(yt) / len(batch))
                loss.backward()
                batch_loss += loss.item()
            opt.step()
            total += batch_loss * len(batch)
        if verbose:
            print(f"    época {ep+1}/{epochs}  pérdida {total/len(windows):.4f}")
    detector.model.eval()
    return detector
