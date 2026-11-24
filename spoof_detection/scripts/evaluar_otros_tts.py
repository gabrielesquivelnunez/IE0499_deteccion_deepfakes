"""
evaluar_otros_tts.py

Entregable 4: generalización del Detector A (LFCC + SVM) a generadores de voz
que NO se usaron al entrenar.

Protocolo (leave-one-speaker-out):
  - Entrenamiento: audios de --raw-dir (reales + sintéticos XTTS v2) de todos los
    hablantes menos el evaluado.
  - Prueba del hablante evaluado: sus reales, sus sintéticos XTTS (control, mismo
    generador del entrenamiento) y sus sintéticos de otros generadores
    (--otros-tts-dir/<generador>/spkNNN_sintetico_<generador>_N.wav).
  - Las ventanas se puntúan con salto --window-sec * (1 - --overlap) y se pasan por
    las estrategias de decisión temporal de streaming/decision.py.

Por generador se reporta:
  - EER por ventana contra las ventanas reales de todos los hablantes.
  - % de ventanas sobre el umbral.
  - Tasa de detección y de falsas alarmas por flujo y tiempo hasta la primera
    decisión, para cada estrategia pedida.
Los reales son los mismos en todas las filas, de modo que la tasa de falsas alarmas
solo depende de la estrategia.

Con --entrenar-con qwen3 (por ejemplo) los audios de ese generador de los demás
hablantes se suman al entrenamiento; esa fila se marca con * en la salida (el
hablante evaluado sigue sin verse, pero el generador sí) y los demás generadores
quedan como prueba de generalización a un sistema no visto.

Uso (desde spoof_detection/, con el entorno del proyecto activo):
    python -m scripts.evaluar_otros_tts --raw-dir $RAW --otros-tts-dir data/otros_tts
    python -m scripts.evaluar_otros_tts --raw-dir $RAW --entrenar-con qwen3 --tag train_qwen3

Escribe results/otros_tts_A[_<tag>].csv y results/puntajes_otros_tts_A[_<tag>]_<w>s.json.
"""

import argparse
import csv
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.audio_io import load_audio_16k
from features.lfcc import lfcc_window_features
from features.windowing import make_windows
from metrics.metrics import compute_eer
from streaming.decision import default_strategies, summarize_strategy

warnings.filterwarnings("ignore", category=FutureWarning)

SPK_RE = re.compile(r"^(spk[0-9]+)", re.IGNORECASE)


def parse_raw(raw_dir):
    """Devuelve [(path, speaker, label)] con label 0 = real, 1 = sintético (XTTS)."""
    out = []
    for p in sorted(Path(raw_dir).glob("*.wav")):
        m = SPK_RE.match(p.stem)
        if not m:
            print(f"  se omite (sin prefijo de hablante): {p.name}")
            continue
        low = p.stem.lower()
        if any(k in low for k in ("sintetico", "fake", "xtts")):
            label = 1
        elif any(k in low for k in ("real", "_ref", "_test")):
            label = 0   # spk001/spk002 usan spkNNN_ref1, _ref2, _test1, _test2 (voces reales)
        else:
            print(f"  se omite (no se reconoce la clase): {p.name}")
            continue
        out.append((str(p), m.group(1).lower(), label))
    return out


def parse_otros(otros_dir, generadores=None):
    """Devuelve {generador: [(path, speaker)]} para cada subcarpeta de otros_dir."""
    res = {}
    base = Path(otros_dir)
    if not base.exists():
        return res
    for sub in sorted(d for d in base.iterdir() if d.is_dir()):
        if generadores and sub.name not in generadores:
            continue
        items = []
        for p in sorted(sub.glob("*.wav")):
            m = SPK_RE.match(p.stem)
            if m:
                items.append((str(p), m.group(1).lower()))
        if items:
            res[sub.name] = items
    return res


def feats(path, window_sec, hop_sec):
    audio, sr = load_audio_16k(path)
    wins = make_windows(audio, sr, window_sec, hop_sec)
    if not wins:
        return np.zeros((0, 0))
    return np.array([lfcc_window_features(w, sr) for w in wins])


def train_clf(X, y):
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=0))
    clf.fit(X, y)
    return clf


def fmt(x, pct=False, nd=2):
    if x is None or x != x:
        return "n/a"
    return f"{x * 100:.1f}" if pct else f"{x:.{nd}f}"


def run_config(raw, otros, window_sec, overlap, strategies, threshold, train_gens=()):
    hop_sec = window_sec * (1.0 - overlap)
    print(f"\n##### ventana {window_sec} s | solape {overlap:.0%} | salto {hop_sec:.2f} s #####")

    # características de entrenamiento (ventanas sin solape) y de prueba (con salto)
    train_feats = {p: feats(p, window_sec, window_sec) for p, _, _ in raw}
    test_feats = {p: feats(p, window_sec, hop_sec) for p, _, _ in raw}
    for g, items in otros.items():
        for p, _ in items:
            test_feats[p] = feats(p, window_sec, hop_sec)
            if g in train_gens:
                train_feats[p] = feats(p, window_sec, window_sec)

    speakers = sorted({s for _, s, _ in raw})
    scores = {"real": [], "xtts": []}
    scores.update({g: [] for g in otros})
    paths = {k: [] for k in scores}

    for held in speakers:
        tr = [(p, l) for p, s, l in raw if s != held and len(train_feats[p])]
        for g in train_gens:
            tr += [(p, 1) for p, s in otros.get(g, []) if s != held and len(train_feats[p])]
        X = np.vstack([train_feats[p] for p, _ in tr])
        y = np.concatenate([[l] * len(train_feats[p]) for p, l in tr])
        if len(set(y)) < 2:
            continue
        clf = train_clf(X, y)

        def score(path):
            f = test_feats[path]
            return clf.predict_proba(f)[:, 1] if len(f) else None

        for p, s, l in raw:
            if s != held:
                continue
            sc = score(p)
            if sc is None:
                continue
            key = "xtts" if l == 1 else "real"
            scores[key].append(sc)
            paths[key].append(p)
        for g, items in otros.items():
            for p, s in items:
                if s != held:
                    continue
                sc = score(p)
                if sc is not None:
                    scores[g].append(sc)
                    paths[g].append(p)
        print(f"  hablante {held} listo")

    real_win = np.concatenate(scores["real"])
    real_streams = [(s, 0) for s in scores["real"]]
    rows = []
    print(f"\nflujos reales: {len(real_streams)} | ventanas reales: {len(real_win)}")
    print(f"{'generador':10s} {'flujos':>6s} {'EER_vent%':>9s} {'vent>umbral%':>12s}  "
          f"{'estrategia':22s} {'detec%':>7s} {'alarma%':>8s} {'t_1ra(s)':>8s} {'F1':>5s}")

    for g in ["xtts"] + list(otros):
        if not scores[g]:
            continue
        gname = g + ("*" if g in train_gens else "")
        gw = np.concatenate(scores[g])
        eer, _ = compute_eer(real_win, gw)
        over = float((gw >= threshold).mean())
        streams = real_streams + [(s, 1) for s in scores[g]]
        for name, make in strategies.items():
            m = summarize_strategy(streams, make, window_sec, hop_sec)
            print(f"{gname:10s} {len(scores[g]):6d} {fmt(eer, True):>9s} {fmt(over, True):>12s}  "
                  f"{name:22s} {fmt(m['detect_rate'], True):>7s} "
                  f"{fmt(m['false_alarm_rate'], True):>8s} {fmt(m['mean_t_first']):>8s} "
                  f"{fmt(m['f1']):>5s}")
            rows.append({"detector": "A", "window_sec": window_sec, "overlap": overlap,
                         "generador": g, "visto_en_entrenamiento": g in train_gens,
                         "entrenado_con": "+".join(["xtts"] + list(train_gens)), "n_flujos": len(scores[g]),
                         "eer_ventana": round(eer, 4), "frac_ventanas_sobre_umbral": round(over, 4),
                         "estrategia": name, "detect_rate": m["detect_rate"],
                         "false_alarm_rate": m["false_alarm_rate"],
                         "mean_t_first": m["mean_t_first"], "f1": m["f1"],
                         "threshold": threshold})

    dump = {k: {Path(p).name: s.tolist() for p, s in zip(paths[k], scores[k])} for k in scores}
    return rows, dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="reales + sintéticos XTTS (entrenamiento)")
    ap.add_argument("--otros-tts-dir", default="data/otros_tts",
                    help="carpeta con una subcarpeta por generador (qwen3, audio8, ...)")
    ap.add_argument("--generadores", nargs="*", default=None)
    ap.add_argument("--window-secs", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--estrategias", nargs="+",
                    default=["ventana_independiente", "promedio_N5", "voto_N5", "sostenido_k3"])
    ap.add_argument("--entrenar-con", nargs="*", default=[],
                    help="generadores de --otros-tts-dir que se suman al entrenamiento")
    ap.add_argument("--tag", default="", help="sufijo para los archivos de salida")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    raw = parse_raw(args.raw_dir)
    if not raw:
        sys.exit(f"No se encontraron .wav válidos en {args.raw_dir}")
    otros = parse_otros(args.otros_tts_dir, args.generadores)
    print(f"Entrenamiento/control: {len(raw)} audios, {len({s for _, s, _ in raw})} hablantes")
    for g, items in otros.items():
        print(f"  {g}: {len(items)} audios")
    if not otros:
        sys.exit(f"No hay subcarpetas con audios en {args.otros_tts_dir}")

    todas = default_strategies(args.threshold)
    faltan = [e for e in args.estrategias if e not in todas]
    if faltan:
        sys.exit(f"Estrategias desconocidas: {faltan}. Disponibles: {list(todas)}")
    strategies = {k: todas[k] for k in args.estrategias}

    train_gens = tuple(args.entrenar_con)
    desconocidos = [g for g in train_gens if g not in otros]
    if desconocidos:
        sys.exit(f"--entrenar-con: no hay audios de {desconocidos} en {args.otros_tts_dir}")
    if train_gens:
        print(f"Entrenamiento: XTTS + {', '.join(train_gens)} (los marcados con * se vieron al entrenar)")

    suf = f"_{args.tag}" if args.tag else ""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for w in args.window_secs:
        r, dump = run_config(raw, otros, w, args.overlap, strategies, args.threshold, train_gens)
        rows += r
        with open(out_dir / f"puntajes_otros_tts_A{suf}_{w:g}s.json", "w") as f:
            json.dump(dump, f)

    out_csv = out_dir / f"otros_tts_A{suf}.csv"
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nResultados guardados en {out_csv}")


if __name__ == "__main__":
    main()
