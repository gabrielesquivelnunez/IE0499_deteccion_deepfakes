# Módulo de detección de deepfakes de audio

Módulo del proyecto **"Detección en tiempo real de deepfakes de audio (voces
sintéticas) aplicada al español costarricense"** (IE-0499 / IE-0435).
Corresponde al **segundo entregable**: detector base y procesamiento por
ventanas.

Integra dos enfoques complementarios, según lo definido en el informe de
revisión:

- **Detector A** (`models/detector_a_lfcc_svm.py`): características LFCC +
  clasificador SVM. Interpretable, liviano, apto para tiempo real.
- **Detector B** (`models/detector_b_aasist.py`): AASIST preentrenado, usado
  por transferencia (transfer learning).

Este módulo está pensado para integrarse dentro de la estructura del
repositorio del laboratorio (`voice-similarity-eval`), respetando su
convención de carpetas (`features/`, `metrics/`, etc.).

## Estructura

```
spoof_detection/
├── data/
│   └── prepare_dataset.py     # organiza train/val/test sin fuga de info. por locutor
├── features/
│   ├── windowing.py           # corta audio en ventanas sucesivas (0.5, 1, 2, 3, 5 s...)
│   └── lfcc.py                # extracción de LFCC (numpy/scipy, sin dependencias pesadas)
├── metrics/
│   └── metrics.py             # EER, precision, recall, F1
├── models/
│   ├── detector_a_lfcc_svm.py # entrena/evalúa el Detector A
│   └── detector_b_aasist.py   # carga AASIST preentrenado, fine-tuning
├── scripts/
│   └── setup_aasist.sh        # clona el repo oficial de AASIST y sus pesos
├── examples/
│   └── run_smoke_test.py      # prueba de humo con audio sintético (no habla real)
├── requirements.txt
└── .gitignore
```

## Instalación

```bash
pip install -r requirements.txt
```

El Detector B necesita además PyTorch, que **no** está en `requirements.txt`
por su tamaño. Instalarlo según tu hardware:

```bash
# CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
# GPU (ajustar la versión de CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Paso 0 — Verificar que todo funciona (smoke test)

Antes de conectar datos reales, correr:

```bash
python examples/run_smoke_test.py
```

Esto genera audio sintético de juguete (tonos, no habla real) y corre el
pipeline completo del Detector A de punta a punta. Si termina sin errores,
el entorno está listo.

## Paso 1 — Organizar el dataset real

Colocar los audios del laboratorio con esta estructura (ajustar
`data/prepare_dataset.py::default_group_extractor` si la convención de
nombres real es distinta):

```
data/raw/
├── real/
│   ├── speaker01_utt001.wav
│   └── ...
└── fake/
    ├── ttsA_speaker01_utt001.wav
    └── ...
```

Luego:

```bash
python data/prepare_dataset.py
```

Esto genera `data/splits/{train,val,test}.csv`, garantizando que ningún
locutor quede repartido entre particiones (evita fuga de información).

## Paso 2 — Entrenar y evaluar el Detector A (LFCC + SVM)

```bash
python -m models.detector_a_lfcc_svm \
    --train data/splits/train.csv \
    --val data/splits/val.csv \
    --test data/splits/test.csv \
    --window-sec 1.0 --hop-sec 1.0
```

Repetir con distintos `--window-sec` (0.5, 1, 2, 3, 5) para la comparación
de longitudes de ventana que pide el entregable.

## Paso 3 — Preparar el Detector B (AASIST)

```bash
bash scripts/setup_aasist.sh          # clona el repo oficial + pesos preentrenados
python -m models.detector_b_aasist    # prueba rápida de carga del modelo
```

El fine-tuning real sobre los datos del laboratorio se conecta llamando a
`AASISTDetector.unfreeze_for_finetuning()` y entrenando con un loop de
PyTorch estándar (pendiente de implementar una vez que el dataset esté
organizado — es el siguiente paso natural tras validar el Detector A).

## Limitaciones conocidas (para documentar en el informe)

- AASIST espera audio a 16 kHz y de duración fija (~4.04 s, `nb_samp` en
  `external/aasist/config/AASIST.conf`); nuestras ventanas de duración
  variable se ajustan (recorte/repetición) a esa longitud, lo cual es una
  limitación a discutir frente al Detector A, que sí acepta ventanas de
  cualquier duración.
- El split actual etiqueta cada ventana con la etiqueta del archivo
  completo del que proviene; falta evaluar si conviene una estrategia más
  fina para ventanas que caen justo en zonas de transición (relevante sobre
  todo se explora la extensión de audio parcialmente falso).
