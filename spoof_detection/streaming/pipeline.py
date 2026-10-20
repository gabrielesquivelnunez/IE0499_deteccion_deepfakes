"""
pipeline.py

Pipeline de detección en flujo continuo:

    chunks de audio --> buffer --> ventanas (window_sec, salto hop_sec)
        --> detector (score_fn) --> estrategia temporal --> decisión real/falso

Es independiente del detector: recibe cualquier función score_fn(ventana, sr)
que devuelva P(sintético). Se alimenta con chunks de tamaño arbitrario, igual
que llegarían de un micrófono o de la reproducción de un archivo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from streaming.decision import TemporalDecider


@dataclass
class StreamEvent:
    window_index: int
    t_end: float               # segundos de audio recibidos al cerrar la ventana
    score: float               # P(sintético) de la ventana
    ready: bool                # ¿la estrategia ya puede decidir?
    is_fake: Optional[bool]    # decisión actual (None si no ready)
    latency_s: float           # tiempo de cómputo del detector para esta ventana


class StreamingDetector:
    def __init__(self, score_fn: Callable[[np.ndarray, int], float], sr: int,
                 window_sec: float, hop_sec: float, decider: TemporalDecider):
        self.score_fn = score_fn
        self.sr = sr
        self.window_len = int(round(window_sec * sr))
        self.hop_len = int(round(hop_sec * sr))
        if self.window_len <= 0 or self.hop_len <= 0:
            raise ValueError("window_sec y hop_sec deben ser positivos")
        self.decider = decider
        self.reset()

    def reset(self):
        self._buf = np.zeros(0, dtype=np.float32)
        self._consumed = 0          # muestras ya descartadas del buffer
        self._index = 0
        self.decider.reset()

    def push(self, chunk: np.ndarray) -> list[StreamEvent]:
        """Agrega un chunk y devuelve un evento por cada ventana completada."""
        self._buf = np.concatenate([self._buf, np.asarray(chunk, dtype=np.float32)])
        events = []
        while len(self._buf) >= self.window_len:
            window = self._buf[: self.window_len]
            t0 = time.perf_counter()
            score = float(self.score_fn(window, self.sr))
            latency = time.perf_counter() - t0

            d = self.decider.update(score)
            t_end = (self._consumed + self.window_len) / self.sr
            events.append(StreamEvent(self._index, t_end, score, d.ready, d.is_fake, latency))

            self._index += 1
            self._buf = self._buf[self.hop_len:]
            self._consumed += self.hop_len
        return events


def stream_audio(detector: StreamingDetector, audio: np.ndarray, chunk_sec: float = 0.1,
                 realtime: bool = False) -> list[StreamEvent]:
    """Reproduce un arreglo de audio como flujo, en chunks de chunk_sec.
    Con realtime=True espera entre chunks para simular la llegada en vivo
    (si el cómputo es más lento que el audio, se acumula retraso)."""
    detector.reset()
    chunk_len = max(1, int(round(chunk_sec * detector.sr)))
    events = []
    start = time.perf_counter()
    for i in range(0, len(audio), chunk_len):
        events += detector.push(audio[i:i + chunk_len])
        if realtime:
            due = start + (i + chunk_len) / detector.sr
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
    return events
