# Resultados — Entregable 2 (corregidos) y exploración del Entregable 3

## Corrección importante (11 oct 2026)
Los resultados iniciales del Detector A (EER 3.47 / 3.99 / 0.00 / 13.75 / 0.00 %) estaban
**contaminados por una confusión de tasa de muestreo**: los audios reales están a 16 kHz y los
sintéticos (XTTS) a 24 kHz. El banco de filtros LFCC cubre 0–sr/2, así que el detector podía separar
las clases solo por la tasa de muestreo. Se comprobó con un test placebo
(`scripts/placebo_sample_rate.py`): con clases idénticas salvo la tasa de muestreo, el pipeline
viejo da EER 0.00 % y el corregido ~50 % (azar).

| Ventana | Placebo, pipeline viejo | Placebo, todo a 16 kHz |
|---|---|---|
| 0.5 s | 0.00 % | 51.43 % |
| 1.0 s | 0.00 % | 48.44 % |
| 2.0 s | 0.00 % | 44.27 % |

Corrección: todo el audio se remuestrea a 16 kHz antes de extraer características
(`features/audio_io.py`). El Detector B ya lo hacía (AASIST exige 16 kHz).

## EER por ventana (corregido; 14 archivos, 4 locutores)
| Ventana | Ventanas reales/sint. | Detector A (LFCC+SVM, LOSO) | Detector B (AASIST-L, zero-shot) |
|---|---|---|---|
| 0.5 s | 143 / 58 | 23.96 % | 41.32 % |
| 1 s | 68 / 28 | 21.74 % | 35.50 % |
| 2 s | 32 / 13 | 15.50 % | 46.51 % |
| 3 s | 20 / 8 | 25.00 % | 50.00 % |
| 5 s | 10 / 5 | 60.00 % | 60.00 % |

## Interpretación
- Detector A (LFCC+SVM): con la confusión eliminada, el EER pasa a 15–25 % para ventanas de 0.5–3 s.
  Hay señal real, pero modesta.
- Detector B (AASIST-L): sin ajuste fino está cerca del azar (35–60 %). Es lo esperado: AASIST-L se
  entrenó con ASVspoof y no con voces XTTS en español. Requiere fine-tuning.
- A 5 s hay solo 10 + 5 ventanas: el 60 % es ruido estadístico, no una tendencia.

## Estrategias temporales (Detector A, LOSO; 10 flujos reales, 4 sintéticos)
Las estrategias que integran varias ventanas reducen las falsas alarmas (de 40–60 % con decisión
independiente a 0–10 %) a costa de mayor tiempo hasta la decisión (1–6 s). `sostenido_k2` con
ventana de 2 s dio la mejor combinación observada (acc 92–100 %, 0 % falsas alarmas, ~3–4 s).
**No es una conclusión estadística:** con solo 4 flujos sintéticos, un flujo cambia el recall 25 puntos.

## Limitaciones
1. Dataset minúsculo (14 archivos, 4 locutores, un solo TTS): resultados exploratorios.
2. Aun con 16 kHz, la cadena de grabación difiere entre clases (reales grabados por personas,
   sintéticos de XTTS). El detector podría estar captando canal/micrófono y no "voz sintética".
   Mitigación pendiente: pasar ambas clases por la misma cadena (mismo códec/ruido) y más TTS.
3. Detector B no ha sido ajustado; falta el bucle de fine-tuning.
4. Umbral fijo de 0.5, no calibrado.
