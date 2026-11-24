# Detección en tiempo real de deepfakes de audio (voces sintéticas)
### Aplicada al español costarricense

Proyecto Eléctrico (IE-0499) — Escuela de Ingeniería Eléctrica, Universidad de Costa Rica.

**Estudiante:** Gabriel Esquivel Núñez (C22799)
**Profesor guía:** Marvin Coto

## Descripción

Sistema que decide si una voz es real o sintética sobre audio que llega de forma continua. El audio se
acumula en un *buffer*, se divide en ventanas, cada ventana se puntúa con un detector y los puntajes se
combinan con una estrategia de decisión temporal.

Detectores:

- **Detector A:** coeficientes LFCC por ventana y SVM con núcleo RBF.
- **Detector B:** AASIST-L preentrenado, con ajuste fino opcional de sus últimas capas.

La evaluación es *leave-one-speaker-out* (LOSO): el hablante evaluado no interviene en el entrenamiento.

**Demostrador:** `spoof_detection/scripts/demo.py` analiza un archivo (con reproducción) o el micrófono y
muestra en una ventana la evolución del puntaje por ventana, el umbral y la decisión (voz real o sintética).
Usa el Detector A, ventanas de 1 s con salto de 0,5 s y la estrategia de umbral sostenido (k=3).

## Estructura del repositorio

```
.
├── docs/                 # Informes en PDF (revisión y diseño, informe final)
├── spoof_detection/      # Código: detectores, pipeline en flujo continuo, demostrador y experimentos
├── README.md
└── RESULTS.md            # Resultados de los entregables 3 y 4
```

Dentro de `spoof_detection/`:

```
data/         lectura del conjunto y particiones por hablante
features/     lectura de audio a 16 kHz, ventanas y LFCC
models/       detectores A y B y ajuste fino de AASIST-L
streaming/    pipeline de flujo continuo y estrategias de decisión
metrics/      EER y métricas por flujo
scripts/      experimentos, demostrador, evaluación con otros generadores y HISPASpoof, micrófono y latencia
generacion/   generación de audios sintéticos (XTTS v2, Qwen3-TTS, Audio8-TTS) y transcripción
external/     código y pesos originales de AASIST (con su licencia)
results/      resultados en CSV/JSON y figuras del demostrador
tests/        pruebas del pipeline y de las estrategias
textos/       textos de lectura y de generación de sintéticos
```

## Estado de los entregables

1. Informe de revisión y diseño del sistema de detección en tiempo real — 4 de setiembre de 2026.
2. Detector base y procesamiento por ventanas — 25 de setiembre de 2026.
3. Sistema de detección en tiempo real e informe de evaluación — 24 de octubre de 2026.
4. Informe final y demostrador funcional — 27 de noviembre de 2026 (informe, demostrador y video listos para entrega).

## Resultados del entregable 3

Conjunto: 11 hablantes y 57 audios (24 reales y 33 sintéticos generados con XTTS v2).

| Ventana | EER A (LFCC+SVM) | EER B sin ajuste | EER B ajustado |
|---|---|---|---|
| 1 s | 19,7 % | 43,0 % | 22,7 % |
| 2 s | 19,5 % | 35,5 % | 22,5 % |
| 4 s | | 29,4 % | 25,0 % |

Con la estrategia de umbral sostenido (k=3), el Detector A con ventanas de 1 s y 50 % de solape decide a
los 2,25 s con 16,7 % de falsas alarmas y 87,9 % de detección. La latencia por ventana es menor que
2 ms para el Detector A en CPU y de unos 19 ms para el Detector B en GPU, con RTF máximo de 0,56 en CPU y
0,075 en GPU. El detalle, las correcciones realizadas y las limitaciones están en [`RESULTS.md`](RESULTS.md).

## Resultados del entregable 4

El detector recomendado es el Detector A (ventana de 1 s, salto de 0,5 s, umbral 0,5, sostenido k=3). Sobre
XTTS v2 detecta 87,9 % de los flujos sintéticos con 16,7 % de falsas alarmas y decide a los 2,25 s.

Generalización a generadores no vistos (Detector A entrenado con XTTS v2 y los reales, LOSO):

| Generador | Detección (1 s, k=3) | EER por ventana (1 s) |
|---|---|---|
| XTTS v2 (control, visto) | 87,9 % | 19,3 % |
| Audio8-TTS | 16,7 % | 46,0 % |
| Qwen3-TTS | 23,5 % | 61,8 % |

En la partición de validación de HISPASpoof (sin entrenar con ella) el detector reconoce XTTS v2 (EER de
10,4 %) pero no ElevenLabs, F5-TTS ni YourTTS (EER de 44 a 51 %); 7,2 % de falsas alarmas en voces reales.
Conclusión: el detector es válido para el generador con que se entrenó; ampliarlo a otros es trabajo futuro.
Detalle en [`RESULTS.md`](RESULTS.md).

## Uso

Instalación, ejecución de experimentos y detección en vivo: ver
[`spoof_detection/README.md`](spoof_detection/README.md).

## Datos

El audio del conjunto (reales y sintéticos), los audios propios en `results/` y los consentimientos informados no se versionan. Los archivos esperados son WAV a
16 kHz con el formato `spkNNN_<descripcion>.wav`, donde el prefijo identifica al hablante.

## Limitaciones conocidas

- Un solo generador de voz para entrenar (XTTS v2) y 11 hablantes; no generaliza a Qwen3, Audio8, ElevenLabs, F5-TTS ni YourTTS.
- El canal de grabación puede estar confundido con la clase (reales grabados con iPhone, sintéticos de
  XTTS).
- Detección de actividad de voz solo en el demostrador (umbral de energía simple); los scripts de evaluación no la usan.
