# Módulo de detección de deepfakes de audio

Módulo del proyecto **"Detección en tiempo real de deepfakes de audio (voces
sintéticas) aplicada al español costarricense"** (IE-0499 / IE-0435).

Decide si una voz es real o sintética sobre audio que llega de forma continua:
el audio se acumula en un *buffer*, se divide en ventanas, cada ventana se
puntúa con un detector y los puntajes se combinan con una estrategia de
decisión temporal.

Detectores:

- **Detector A** (`models/detector_a_lfcc_svm.py`, `models/detector_a_stream.py`):
  características LFCC y clasificador SVM con núcleo RBF. Liviano e
  interpretable.
- **Detector B** (`models/detector_b_aasist.py`): AASIST-L preentrenado, con
  ajuste fino opcional de sus últimas capas (`models/finetune_aasist.py`).

Este módulo está pensado para integrarse en el repositorio del laboratorio
(`voice-similarity-eval`), respetando su convención de carpetas.

## Estructura

```
spoof_detection/
├── data/                         # lectura del conjunto y particiones por locutor
│   └── prepare_dataset.py
├── features/                     # entrada de audio y características
│   ├── audio_io.py               # lectura y remuestreo a 16 kHz mono
│   ├── windowing.py              # ventanas sucesivas (0.5, 1, 2, 3, 5 s...)
│   └── lfcc.py                   # extracción de LFCC (numpy/scipy)
├── models/                       # detectores
│   ├── detector_a_lfcc_svm.py    # Detector A
│   ├── detector_a_stream.py      # Detector A para flujo continuo
│   ├── detector_b_aasist.py      # AASIST-L preentrenado
│   └── finetune_aasist.py        # ajuste fino de las últimas capas
├── streaming/                    # flujo continuo
│   ├── pipeline.py               # buffer, ventanas, detector y decisión
│   └── decision.py               # estrategias de decisión temporal y métricas por flujo
├── metrics/
│   └── metrics.py                # EER, precisión, recall, F1
├── scripts/                      # experimentos y demostrador (ver más abajo)
├── generacion/                   # generación de audios sintéticos y transcripción
│   ├── generar_sinteticos_texto.py   # XTTS v2 (generador de entrenamiento)
│   ├── generar_sinteticos_qwen3.py   # Qwen3-TTS (generador de prueba)
│   ├── generar_sinteticos_audio8.py  # Audio8-TTS (generador de prueba)
│   └── transcribir_cortos.py         # transcribe los audios cortos con faster-whisper
├── textos/                       # textos de lectura y de generación de sintéticos
├── external/aasist/              # código y pesos originales de AASIST (con su licencia)
├── results/                      # resultados (CSV, JSON) y figuras del demostrador
├── tests/                        # pruebas del pipeline y de las estrategias
├── examples/
│   └── run_smoke_test.py         # prueba de humo con audio sintético
└── requirements.txt
```

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

El Detector B necesita además PyTorch, que no está en `requirements.txt` por
su tamaño:

```bash
# CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
# GPU (ajustar la versión de CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Para la detección con micrófono:

```bash
pip install sounddevice
sudo apt install libportaudio2
```

Para el demostrador con archivo y gráfico: `sudo apt install python3-tk ffmpeg` (interfaz gráfica de
matplotlib y `ffplay` para reproducir el archivo).

Para generar sintéticos se usa XTTS v2 en un entorno aparte (`coqui-tts`, con
`COQUI_TOS_AGREED=1`). La transcripción de los audios cortos usa
`faster-whisper`.

## Prueba de humo

```bash
python examples/run_smoke_test.py
```

Genera audio de juguete (tonos) y corre el pipeline del Detector A de punta a
punta. Si termina sin errores, el entorno está listo. Las pruebas automáticas
se ejecutan con:

```bash
python -m pytest tests
```

## Datos

El audio no se versiona. Los comandos reciben la carpeta con `--raw-dir`
(por ejemplo `RAW=ruta/al/conjunto/data/raw`).

- Archivos WAV con el prefijo del hablante: `spk001_...wav`, `spk002_...wav`.
  El prefijo se usa como identificador de locutor en la validación
  *leave-one-speaker-out*. La regla que asigna la etiqueta real o falso está
  en `data/prepare_dataset.py::scan_dataset`.
- Todo el audio se convierte a 16 kHz mono al leerlo (`features/audio_io.py`),
  de modo que la frecuencia de muestreo no distingue las clases.
- Los consentimientos informados de los participantes se guardan fuera del
  repositorio.

Los comandos siguientes se ejecutan desde `spoof_detection/` con el entorno
activo.

## Experimentos

Detector A y Detector B sin ajuste, EER por ventana:

```bash
python -m scripts.run_entregable2_experiments --raw-dir $RAW
```

Estrategias de decisión temporal con el Detector A:

```bash
python -m scripts.run_temporal_strategies --raw-dir $RAW
```

Ajuste fino de AASIST-L con *leave-one-speaker-out*, umbral calibrado por
hablante y puntajes por ventana (`results/puntajes_B_<w>s.json`):

```bash
python -u -m scripts.run_finetune_loso --raw-dir $RAW --epochs 8 --window-secs 1 2 4
```

Los resultados se escriben en `results/`. La corrida con las tres ventanas
tarda unas dos horas en una GPU de 4 GB. Otros argumentos: `--overlaps`,
`--lr`, `--threshold`, `--device`, `--out-prefix`.

Latencia y RTF:

```bash
python -m scripts.measure_latency --help
```

Prueba placebo de tasa de muestreo (clases idénticas salvo la frecuencia de
muestreo; el EER debe ser cercano a 50 %):

```bash
python -m scripts.placebo_sample_rate --help
```

## Detección en flujo continuo

Reproduce un archivo como flujo y muestra el puntaje y la decisión por
ventana:

```bash
python -m scripts.simulate_stream --wav archivo.wav --raw-dir $RAW \
    --detector a --window-sec 2 --overlap 0.5 --strategy sostenido_k2
```

Captura del micrófono con el mismo pipeline:

```bash
python -m scripts.live_mic --list-devices
python -m scripts.live_mic --raw-dir $RAW --detector a --window-sec 2 \
    --overlap 0.5 --strategy sostenido_k2 --seconds 30 \
    --save-wav results/vivo.wav --out results/vivo.csv
```

Estrategias disponibles: `ventana_independiente`, `promedio_N3`,
`promedio_N5`, `voto_N3`, `voto_N5`, `sostenido_k2`, `sostenido_k3`.
`--seconds 0` captura hasta Ctrl+C y `--device N` elige el micrófono.

## Demostrador

Muestra la evolución del puntaje, el umbral y la decisión (VOZ REAL / VOZ SINTÉTICA / ESPERANDO VOZ) mientras
suena o llega el audio. Usa el Detector A, entrenado con todos los audios de `--raw-dir` (la primera vez
tarda; se guarda en `models/checkpoints/`).

```bash
# archivo (se reproduce por los parlantes y se analiza como flujo)
python -m scripts.demo --raw-dir $RAW --wav ruta/audio.wav
# micrófono en vivo (--device N elige el micrófono)
python -m scripts.demo --raw-dir $RAW --mic
# sin ventana, para guardar una figura
python -m scripts.demo --raw-dir $RAW --wav ruta/audio.wav --no-audio --headless --snapshot results/demo.png
```

Opciones: `--window-sec`, `--overlap`, `--strategy`, `--threshold`, `--vad-db` (nivel de silencio en dBFS) y
`--span` (segundos visibles).

## Generalización a otros generadores

Qwen3-TTS y Audio8-TTS se usan solo para generar audios de prueba (clonan al hablante desde su audio corto y
su transcripción); no se entrenan. Se instalan en entornos aparte y cada script escribe en
`data/otros_tts/<generador>/spkNNN_sintetico_<generador>_N.wav`:

```bash
python generacion/generar_sinteticos_qwen3.py --help
python generacion/generar_sinteticos_audio8.py --help
```

Evaluación del Detector A con XTTS (control) y los generadores no vistos, con validación dejando un hablante
fuera (`--entrenar-con qwen3 --tag train_qwen3` los suma al entrenamiento):

```bash
python -m scripts.evaluar_otros_tts --raw-dir $RAW --otros-tts-dir data/otros_tts
```

Evaluación sin entrenar con la partición de validación de HISPASpoof
(`purdueviperlab/HISPASpoof` en Hugging Face, `detection/val.tar` y `detection/protocols/val_metadata.csv`):

```bash
python -m scripts.evaluar_hispaspoof --raw-dir $RAW --tar RUTA/val.tar --metadata RUTA/val_metadata.csv
```

Resultados en `results/otros_tts_A*.csv` y `results/hispaspoof_A_val.csv`.

## Generación de sintéticos

```bash
python generacion/generar_sinteticos_texto.py --help
python generacion/transcribir_cortos.py --help
```

`generacion/generar_sinteticos_texto.py` produce, por hablante, un sintético largo con el
texto común (`textos/largo.txt`) y uno corto con la transcripción del audio
corto del hablante. Las transcripciones (`textos/cortos/`) no se versionan.

## Resultados

Ver [`../RESULTS.md`](../RESULTS.md).

## Limitaciones conocidas

- AASIST espera audio a 16 kHz y de duración fija (unos 4 s, 64 600
  muestras); las ventanas de otra duración se recortan o se repiten hasta esa
  longitud. El Detector A acepta ventanas de cualquier duración.
- Cada ventana recibe la etiqueta del archivo del que proviene. El audio
  parcialmente falso, con una sección sustituida por voz sintética, no está
  evaluado.
- Un solo generador de voz para entrenar (XTTS v2) y 11 hablantes. El detector no
  generaliza a Qwen3, Audio8, ElevenLabs, F5-TTS ni YourTTS. El canal de grabación
  puede estar confundido con la clase.
- La detección de actividad de voz (umbral de energía) existe solo en el
  demostrador; los scripts de evaluación no la usan, y sin ella el Detector A
  puede clasificar silencio y ruido de fondo como sintético.
