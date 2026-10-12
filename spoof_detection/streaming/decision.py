"""
decision.py

Estrategias de decisión temporal para el detector en tiempo real.

El detector produce un puntaje (probabilidad de audio sintético) por cada
ventana de audio. Aquí se decide cómo convertir esa secuencia de puntajes
en una decisión real/falso que se va actualizando a medida que llega audio.

Estrategias (una clase por cada una):

1. PerWindow            : cada ventana decide sola, sin memoria.
2. MovingAverage        : promedio de las últimas N probabilidades vs. umbral.
3. MajorityVote         : cada una de las últimas N ventanas "vota" y gana la
                          mayoría estricta (en un empate, se decide "real").
4. SustainedThreshold   : declara "falso" solo si el puntaje se mantiene sobre
                          el umbral durante k ventanas consecutivas, y "real"
                          si se mantiene bajo el umbral k ventanas
                          consecutivas; en cualquier otro caso mantiene la
                          decisión anterior (histéresis).

Todas exponen la misma interfaz:
    decider.reset()
    decision = decider.update(score)    # -> Decision(ready, is_fake, score)

`ready` indica si ya hay suficiente evidencia para emitir una decisión (p. ej.
el promedio de N ventanas necesita N ventanas). Mientras `ready` sea False,
`is_fake` es None. El "tiempo hasta la primera decisión confiable" es el
instante (en segundos de audio consumido) en que `ready` pasa a True por
primera vez.

Además se incluyen funciones para evaluar una estrategia sobre un conjunto de
flujos (un flujo = la secuencia de puntajes de un archivo de audio):
    run_stream(...)           -> línea de tiempo de decisiones de un flujo
    summarize_strategy(...)   -> métricas agregadas (precisión, recall, F1,
                                 cobertura, tiempo hasta decisión, alarmas)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class Decision:
    ready: bool                 # ¿hay evidencia suficiente para decidir?
    is_fake: Optional[bool]     # None mientras ready=False
    score: Optional[float]      # puntaje agregado usado para decidir (si aplica)


class TemporalDecider:
    """Interfaz común de todas las estrategias."""

    name = "base"

    def reset(self) -> None:
        raise NotImplementedError

    def update(self, score: float) -> Decision:
        raise NotImplementedError


class PerWindow(TemporalDecider):
    """Decisión independiente por ventana (sin memoria)."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.name = "ventana_independiente"

    def reset(self) -> None:
        pass

    def update(self, score: float) -> Decision:
        return Decision(ready=True, is_fake=score >= self.threshold, score=score)


class MovingAverage(TemporalDecider):
    """Promedio de las últimas N probabilidades comparado con el umbral."""

    def __init__(self, threshold: float = 0.5, n: int = 3):
        if n < 1:
            raise ValueError("n debe ser >= 1")
        self.threshold = threshold
        self.n = n
        self.name = f"promedio_N{n}"
        self._buf: deque = deque(maxlen=n)

    def reset(self) -> None:
        self._buf.clear()

    def update(self, score: float) -> Decision:
        self._buf.append(score)
        if len(self._buf) < self.n:
            return Decision(ready=False, is_fake=None, score=None)
        mean = float(np.mean(self._buf))
        return Decision(ready=True, is_fake=mean >= self.threshold, score=mean)


class MajorityVote(TemporalDecider):
    """Voto mayoritario entre las últimas N ventanas.

    Cada ventana vota "falso" si su puntaje >= umbral. Gana "falso" solo con
    mayoría estricta (más de N/2 votos); en un empate (N par) se decide "real",
    que es la opción conservadora ante falsas alarmas.
    """

    def __init__(self, threshold: float = 0.5, n: int = 3):
        if n < 1:
            raise ValueError("n debe ser >= 1")
        self.threshold = threshold
        self.n = n
        self.name = f"voto_N{n}"
        self._votes: deque = deque(maxlen=n)

    def reset(self) -> None:
        self._votes.clear()

    def update(self, score: float) -> Decision:
        self._votes.append(score >= self.threshold)
        if len(self._votes) < self.n:
            return Decision(ready=False, is_fake=None, score=None)
        frac_fake = sum(self._votes) / self.n
        return Decision(ready=True, is_fake=sum(self._votes) * 2 > self.n, score=frac_fake)


class SustainedThreshold(TemporalDecider):
    """Umbral sostenido durante k ventanas consecutivas (con histéresis).

    - Declara "falso" cuando hay k ventanas seguidas con puntaje >= umbral.
    - Declara "real" cuando hay k ventanas seguidas con puntaje < umbral.
    - Si no se cumple ninguna de las dos, conserva la decisión anterior.
    Un pico aislado de ruido no cambia la decisión, a costa de tardar al menos
    k ventanas en emitir la primera.
    """

    def __init__(self, threshold: float = 0.5, k: int = 3):
        if k < 1:
            raise ValueError("k debe ser >= 1")
        self.threshold = threshold
        self.k = k
        self.name = f"sostenido_k{k}"
        self.reset()

    def reset(self) -> None:
        self._run_high = 0
        self._run_low = 0
        self._state: Optional[bool] = None

    def update(self, score: float) -> Decision:
        if score >= self.threshold:
            self._run_high += 1
            self._run_low = 0
        else:
            self._run_low += 1
            self._run_high = 0

        if self._run_high >= self.k:
            self._state = True
        elif self._run_low >= self.k:
            self._state = False

        if self._state is None:
            return Decision(ready=False, is_fake=None, score=score)
        return Decision(ready=True, is_fake=self._state, score=score)


def default_strategies(threshold: float = 0.5) -> dict[str, Callable[[], TemporalDecider]]:
    """Conjunto estándar de estrategias a comparar (nombre -> fábrica)."""
    return {
        "ventana_independiente": lambda: PerWindow(threshold),
        "promedio_N3": lambda: MovingAverage(threshold, 3),
        "promedio_N5": lambda: MovingAverage(threshold, 5),
        "voto_N3": lambda: MajorityVote(threshold, 3),
        "voto_N5": lambda: MajorityVote(threshold, 5),
        "sostenido_k2": lambda: SustainedThreshold(threshold, 2),
        "sostenido_k3": lambda: SustainedThreshold(threshold, 3),
    }


# ---------------------------------------------------------------------------
# Evaluación sobre flujos
# ---------------------------------------------------------------------------

def decision_time(window_index: int, window_sec: float, hop_sec: float) -> float:
    """Segundos de audio consumidos cuando termina la ventana `window_index`
    (0-based): la primera ventana termina en window_sec, y cada ventana
    siguiente avanza hop_sec."""
    return window_sec + window_index * hop_sec


def run_stream(scores: Sequence[float], decider: TemporalDecider,
               window_sec: float, hop_sec: float) -> dict:
    """Pasa la secuencia de puntajes de un flujo por la estrategia.

    Returns dict con:
        decisions     : lista de Decision, una por ventana
        first_index   : índice de la ventana de la primera decisión (o None)
        first_is_fake : primera decisión emitida (o None)
        t_first       : segundos de audio hasta la primera decisión (o None)
        ever_fake     : True si en algún momento la decisión fue "falso"
    """
    decider.reset()
    decisions = []
    first_index = None
    first_is_fake = None
    ever_fake = False

    for i, s in enumerate(scores):
        d = decider.update(float(s))
        decisions.append(d)
        if d.ready:
            if first_index is None:
                first_index = i
                first_is_fake = d.is_fake
            if d.is_fake:
                ever_fake = True

    t_first = None if first_index is None else decision_time(first_index, window_sec, hop_sec)
    return {
        "decisions": decisions,
        "first_index": first_index,
        "first_is_fake": first_is_fake,
        "t_first": t_first,
        "ever_fake": ever_fake,
    }


def summarize_strategy(streams: Sequence[tuple], make_decider: Callable[[], TemporalDecider],
                       window_sec: float, hop_sec: float) -> dict:
    """Métricas de una estrategia sobre varios flujos.

    streams: lista de (scores, label) con label 1 = sintético, 0 = real.

    Métricas (clase positiva = sintético; la primera decisión de cada flujo es
    la que se evalúa en accuracy/precision/recall/F1):
        n_streams, coverage     : flujos totales y fracción que llegó a decidir
        accuracy, precision, recall, f1 : sobre los flujos que sí decidieron
        mean_t_first            : segundos promedio hasta la primera decisión
        false_alarm_rate        : fracción de flujos REALES en los que la
                                  decisión fue "falso" en algún momento
        detect_rate             : fracción de flujos SINTÉTICOS en los que la
                                  decisión fue "falso" en algún momento
    """
    decider = make_decider()
    n_real = sum(1 for _, y in streams if y == 0)
    n_fake = sum(1 for _, y in streams if y == 1)

    decided = tp = fp = fn = tn = 0
    times = []
    alarm_real = detect_fake = 0

    for scores, label in streams:
        r = run_stream(scores, decider, window_sec, hop_sec)
        if label == 0 and r["ever_fake"]:
            alarm_real += 1
        if label == 1 and r["ever_fake"]:
            detect_fake += 1

        if r["first_index"] is None:
            continue
        decided += 1
        times.append(r["t_first"])
        pred_fake = bool(r["first_is_fake"])
        if label == 1 and pred_fake:
            tp += 1
        elif label == 1 and not pred_fake:
            fn += 1
        elif label == 0 and pred_fake:
            fp += 1
        else:
            tn += 1

    def ratio(a, b):
        return a / b if b > 0 else float("nan")

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall) > 0
          else float("nan"))

    return {
        "n_streams": len(streams),
        "coverage": ratio(decided, len(streams)),
        "accuracy": ratio(tp + tn, decided),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_t_first": float(np.mean(times)) if times else float("nan"),
        "false_alarm_rate": ratio(alarm_real, n_real),
        "detect_rate": ratio(detect_fake, n_fake),
    }
