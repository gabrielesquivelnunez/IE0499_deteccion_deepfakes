"""
audio_io.py

Carga de audio con tasa de muestreo UNIFICADA (16 kHz, mono).

Por qué existe: en el conjunto actual los audios reales están a 16 kHz y los
sintéticos (XTTS) a 24 kHz. Los LFCC usan un banco de filtros lineal que cubre
0 a sr/2, así que la tasa de muestreo cambia QUÉ frecuencias mide cada filtro y
el detector puede separar las clases solo por eso (confusión etiqueta/sr).
Remuestrear TODO a 16 kHz antes de extraer características la elimina.
(AASIST ya exige 16 kHz.)
"""

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


def resample_to(audio: np.ndarray, sr_in: int, sr_out: int = TARGET_SR) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32)
    g = np.gcd(int(sr_in), int(sr_out))
    return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)


def to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample_to(audio, sr, TARGET_SR)


def load_audio_16k(path):
    """Devuelve (audio mono float32, 16000)."""
    audio, sr = sf.read(path)
    return to_mono_16k(audio, sr), TARGET_SR
