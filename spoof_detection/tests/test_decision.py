"""
Pruebas de streaming/decision.py con secuencias de puntajes escritas a mano,
donde la respuesta correcta se puede razonar a ojo.

Uso (desde spoof_detection/):
    python -m tests.test_decision
(también funciona con pytest si lo tenés instalado)
"""

import math

from streaming.decision import (
    PerWindow, MovingAverage, MajorityVote, SustainedThreshold,
    run_stream, summarize_strategy, decision_time,
)

THR = 0.5


def flags(decider, scores):
    decider.reset()
    return [(d.ready, d.is_fake) for d in (decider.update(s) for s in scores)]


def test_per_window():
    out = flags(PerWindow(THR), [0.1, 0.9, 0.5, 0.49])
    assert out == [(True, False), (True, True), (True, True), (True, False)]


def test_moving_average_needs_n_windows():
    out = flags(MovingAverage(THR, 3), [0.9, 0.9, 0.9, 0.1])
    assert out[0] == (False, None) and out[1] == (False, None)
    assert out[2] == (True, True)            # media 0.9
    assert out[3] == (True, True)            # media (0.9+0.9+0.1)/3 = 0.633


def test_majority_vote():
    out = flags(MajorityVote(THR, 3), [0.9, 0.1, 0.9, 0.1, 0.1])
    assert out[0][0] is False and out[1][0] is False
    assert out[2] == (True, True)            # 2 de 3 votan falso
    assert out[3] == (True, False)           # ventana [0.1, 0.9, 0.1] -> 1 de 3
    assert out[4] == (True, False)


def test_majority_vote_tie_is_real():
    out = flags(MajorityVote(THR, 4), [0.9, 0.9, 0.1, 0.1])
    assert out[3] == (True, False)           # 2 de 4: empate -> real


def test_sustained_threshold_hysteresis():
    out = flags(SustainedThreshold(THR, 2), [0.9, 0.9, 0.1, 0.9, 0.1, 0.1])
    assert out[0] == (False, None)           # aún no hay 2 seguidas
    assert out[1] == (True, True)            # 2 altas seguidas -> falso
    assert out[2] == (True, True)            # 1 baja: se mantiene
    assert out[3] == (True, True)
    assert out[4] == (True, True)            # 1 baja (la racha se reinició)
    assert out[5] == (True, False)           # 2 bajas seguidas -> real


def test_single_spike_on_real_audio():
    # Audio REAL con un solo pico de ruido: la ventana independiente da falsa
    # alarma; las estrategias con memoria no.
    spike = [0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
    assert run_stream(spike, PerWindow(THR), 1.0, 1.0)["ever_fake"] is True
    assert run_stream(spike, MovingAverage(THR, 3), 1.0, 1.0)["ever_fake"] is False
    assert run_stream(spike, MajorityVote(THR, 3), 1.0, 1.0)["ever_fake"] is False
    assert run_stream(spike, SustainedThreshold(THR, 2), 1.0, 1.0)["ever_fake"] is False


def test_time_to_first_decision():
    fake = [0.9] * 8
    # ventana de 2 s, salto de 1 s (50% de solape)
    assert decision_time(0, 2.0, 1.0) == 2.0
    assert run_stream(fake, PerWindow(THR), 2.0, 1.0)["t_first"] == 2.0
    assert run_stream(fake, MovingAverage(THR, 3), 2.0, 1.0)["t_first"] == 4.0   # idx 2 -> 2 + 2*1
    assert run_stream(fake, SustainedThreshold(THR, 3), 2.0, 1.0)["t_first"] == 4.0


def test_stream_too_short_never_decides():
    r = run_stream([0.9, 0.9], MovingAverage(THR, 5), 1.0, 1.0)
    assert r["first_index"] is None and r["t_first"] is None


def test_summarize_metrics():
    streams = [
        ([0.9] * 6, 1),                               # sintético claro
        ([0.8] * 6, 1),                               # sintético claro
        ([0.1] * 6, 0),                               # real claro
        ([0.1, 0.1, 0.9, 0.1, 0.1, 0.1], 0),          # real con un pico
        ([0.9, 0.9], 1),                              # demasiado corto para N=3
    ]
    m = summarize_strategy(streams, lambda: PerWindow(THR), 1.0, 1.0)
    # La primera ventana decide en todos los flujos: 5 decididos.
    assert m["coverage"] == 1.0
    assert m["false_alarm_rate"] == 0.5               # el real con pico dispara alarma
    assert m["detect_rate"] == 1.0

    m3 = summarize_strategy(streams, lambda: MovingAverage(THR, 3), 1.0, 1.0)
    assert math.isclose(m3["coverage"], 4 / 5)        # el flujo corto no alcanza N=3
    assert m3["false_alarm_rate"] == 0.0              # el pico queda promediado
    assert m3["precision"] == 1.0 and m3["recall"] == 1.0
    assert math.isclose(m3["mean_t_first"], 3.0)      # idx 2 -> 1 + 2*1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} pruebas pasaron.")
