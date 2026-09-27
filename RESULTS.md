# Resultados preliminares -- Entregable 2

Experimentos corridos con `scripts/run_entregable2_experiments.py` sobre el
dataset real del laboratorio (`voice-similarity-eval`): 14 archivos, 4
locutores (Daniel, Felipe, Nayelhi, Paula), 10 grabaciones naturales y 4
versiones sintéticas (XTTS).

## Resultados por duración de ventana

| Ventana | Detector A (LFCC+SVM, LOSO) -- EER | Detector B (AASIST-L, zero-shot) -- EER | Ventanas evaluadas (real/fake) |
|---|---|---|---|
| 0.5 s | 3.47% | 41.32% | 143 / 58 |
| 1.0 s | 3.99% | 35.50% | 68 / 28 |
| 2.0 s | 0.00% | 46.51% | 32 / 13 |
| 3.0 s | 13.75% | 50.00% | 20 / 8 |
| 5.0 s | 0.00% | 60.00% | 10 / 5 |

## Interpretación

- **El Detector A (LFCC+SVM) supera claramente al Detector B en esta etapa**,
  lo cual es esperable: A se entrena directamente sobre estos datos mediante
  leave-one-speaker-out (LOSO), mientras que B se evalúa **zero-shot**, es
  decir, usando los pesos de AASIST-L tal como fueron entrenados sobre
  ASVspoof2019 LA (habla en inglés, ataques de TTS/VC distintos a XTTS),
  sin ningún ajuste al español costarricense ni al dataset del laboratorio.
- El desempeño débil de AASIST-L (35-60% EER, cercano al azar) **no es una
  falla del wrapper ni del modelo**: es el punto de partida esperado antes
  del fine-tuning, y confirma la necesidad de la etapa de transferencia
  planeada para el entregable 3.
- **Los resultados de ambos detectores deben tomarse con cautela.** El
  dataset actual es muy pequeño (4 locutores, 14 archivos), por lo que:
  - Los EER de 0% en algunas ventanas probablemente reflejan el tamaño
    reducido del conjunto de prueba en esa ronda de LOSO (p. ej., a 5 s solo
    hay 10 ventanas reales y 5 sintéticas en total), no un desempeño
    perfecto real.
  - No hay una tendencia limpia y monótona entre duración de ventana y EER
    para el Detector A (3.99% -> 0.00% -> 13.75% -> 0.00%), lo cual es
    consistente con ruido estadístico por bajo número de muestras, más que
    con una relación real entre ventana y desempeño.
- **Ampliar el dataset (autorizado por el profesor guía) es el paso que más
  va a mejorar la fiabilidad de estos números**, más que cualquier ajuste
  fino de hiperparámetros en esta etapa.

## Próximos pasos (entregable 3)

1. Fine-tuning de AASIST-L sobre el dataset del laboratorio (actualmente
   solo se evaluó zero-shot).
2. Repetir estos experimentos con un dataset ampliado (más locutores) para
   obtener EER estadísticamente más confiables.
3. Evaluar estrategias de agregación temporal entre ventanas sucesivas
   (promedio, voto mayoritario, umbral sostenido), pendiente en esta etapa.