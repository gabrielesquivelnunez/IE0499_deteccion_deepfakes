spoof_detection/
├── data/
│   └── prepare_dataset.py        # lectura del conjunto y particiones por locutor
├── features/
│   ├── audio_io.py               # lectura y remuestreo a 16 kHz mono
│   ├── windowing.py              # ventanas sucesivas (0.5, 1, 2, 3, 5 s...)
│   └── lfcc.py                   # extracción de LFCC (numpy/scipy)
├── metrics/
│   └── metrics.py                # EER, precisión, recall, F1
├── models/
│   ├── detector_a_lfcc_svm.py    # Detector A
│   ├── detector_a_stream.py      # Detector A para flujo continuo
│   ├── detector_b_aasist.py      # AASIST-L preentrenado
│   └── finetune_aasist.py        # ajuste fino de las últimas capas
├── streaming/
│   ├── pipeline.py               # buffer, ventanas, detector y decisión
│   └── decision.py               # estrategias de decisión temporal y métricas por flujo
├── scripts/                      # experimentos y herramientas (ver más abajo)
├── tests/                        # pruebas del pipeline y de las estrategias
├── examples/
│   └── run_smoke_test.py         # prueba de humo con audio sintético
├── textos/                       # textos de lectura y de generación de sintéticos
├── generar_sinteticos_texto.py   # genera sintéticos con XTTS v2
├── transcribir_cortos.py         # transcribe los audios cortos con faster-whisper
└── requirements.txt