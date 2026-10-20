"""El pipeline en streaming debe dar lo mismo que evaluar la secuencia de puntajes de una vez."""
import numpy as np
from streaming.decision import SustainedThreshold, run_stream
from streaming.pipeline import StreamingDetector, stream_audio
from features.windowing import make_windows

SR = 16000

def fake_score(w, sr):           # puntaje determinista dependiente de la ventana
    return float(np.clip(0.5 + w.mean() * 50, 0, 1))

def make_audio():
    rng = np.random.default_rng(0)
    return (rng.standard_normal(SR * 7) * 0.1 + np.linspace(0, 0.02, SR * 7)).astype(np.float32)

def test_chunk_size_does_not_change_result():
    audio = make_audio()
    res = []
    for chunk in (0.013, 0.1, 1.0, 7.0):
        sd = StreamingDetector(fake_score, SR, 1.0, 0.5, SustainedThreshold(0.5, 2))
        ev = stream_audio(sd, audio, chunk)
        res.append([(e.window_index, round(e.score, 6), e.is_fake) for e in ev])
    assert all(r == res[0] for r in res)

def test_matches_offline_windows():
    audio = make_audio()
    sd = StreamingDetector(fake_score, SR, 2.0, 1.0, SustainedThreshold(0.5, 2))
    ev = stream_audio(sd, audio, 0.1)
    wins = make_windows(audio, SR, 2.0, 1.0)
    assert len(ev) == len(wins)
    off = run_stream([fake_score(w, SR) for w in wins], SustainedThreshold(0.5, 2), 2.0, 1.0)
    assert [e.is_fake for e in ev] == [d.is_fake for d in off["decisions"]]
    assert abs(ev[0].t_end - 2.0) < 1e-9 and abs(ev[1].t_end - 3.0) < 1e-9

if __name__ == "__main__":
    test_chunk_size_does_not_change_result(); print("OK  test_chunk_size_does_not_change_result")
    test_matches_offline_windows(); print("OK  test_matches_offline_windows")
