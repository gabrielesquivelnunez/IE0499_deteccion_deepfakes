# Resultados: entregable 3 (actualizado el 22 de octubre de 2026)

Sustituye a la versión anterior, que correspondía a 14 archivos y 4 locutores y tenía el Detector B con
las clases invertidas.

## Correcciones respecto a versiones anteriores

1. **Tasa de muestreo (11 oct).** Los reales estaban a 16 kHz y los sintéticos a 24 kHz, y el banco de
   filtros de los LFCC cubre de 0 a sr/2, de modo que el Detector A podía separar las clases por la tasa
   de muestreo. Todo el audio se remuestrea ahora a 16 kHz mono (`features/audio_io.py`). La prueba
   placebo (`scripts/placebo_sample_rate.py`) dio EER de 0,00 % con el procesamiento anterior y de 44 a
   51 % con el corregido. Los EER del entregable 2 para el Detector A (0 a 14 %) quedan invalidados.
2. **Índice de clases del Detector B.** AASIST-L usa el índice 1 para *bonafide* y el código lo trataba
   como *spoof*, de modo que los puntajes estaban invertidos. Corregido.
3. **Identificadores de locutor.** La expresión regular de `data/prepare_dataset.py` solo admitía letras
   y `spk001` se agrupaba como `spk`, por lo que la validación *leave-one-speaker-out* dejaba al mismo
   hablante en entrenamiento y prueba. Corregida (`[a-zA-Z0-9]+`); los resultados con 35 grupos se
   descartaron.

## Conjunto de datos

- 11 hablantes (spk001 a spk011), 57 audios: 24 reales y 33 sintéticos (XTTS v2).
- spk005 a spk011: consentimiento firmado, notas de voz de iPhone (ALAC en cinco casos, AAC en spk009 y
  spk010), un audio largo (50 a 70 s, texto común) y uno corto (19 a 26 s, contenido libre).
- Sintéticos por hablante: uno con una frase fija y, en la versión actual, uno largo con el texto común y
  uno corto con la transcripción del audio corto (`faster-whisper`). Para spk001 a spk004 se usaron
  textos genéricos.
- Evaluación: *leave-one-speaker-out* (LOSO), EER sobre las ventanas reunidas.

## EER (%) por ventana

| Ventana | A (LFCC+SVM) | B sin ajuste | B ajustado (LOSO) |
|---|---|---|---|
| 0,5 s | 19,2 | 48,3 | |
| 1 s | 19,7 | 43,0 | 22,7 |
| 2 s | 19,5 | 35,5 | 22,5 |
| 3 s | 19,1 | 30,0 | |
| 4 s | | 29,4 | 25,0 |
| 5 s | 16,7 | 24,9 | |

Ajuste fino: últimas 3 capas de AASIST-L (17 602 parámetros), 8 épocas, `lr` = 1e-4, pérdida ponderada,
ganancia aleatoria de ±6 dB. El umbral calibrado de cada hablante (punto de EER de los demás) quedó entre
0,56 y 0,60.

## Estrategias temporales (57 flujos: 24 reales y 33 sintéticos)

Precisión, *recall* y F1 con la primera decisión de cada flujo (clase positiva: falso). FA: fracción de
flujos reales con alguna decisión "falso". Det: la misma fracción en flujos sintéticos.

**Detector A** (umbral 0,5):

| Config. | Estrategia | t1 (s) | F1 | FA (%) | Det (%) |
|---|---|---|---|---|---|
| 0,5 s / 50 % | independiente | 0,50 | 0,70 | 54,2 | 100 |
| 0,5 s / 50 % | promedio N=5 | 1,50 | 0,79 | 29,2 | 97,0 |
| 1 s / 50 % | voto N=5 | 3,00 | 0,80 | 16,7 | 90,9 |
| 1 s / 50 % | sostenido k=3 | 2,25 | 0,80 | 16,7 | 87,9 |
| 2 s / 50 % | sostenido k=2 | 3,04 | 0,80 | 20,8 | 93,9 |
| 1 s / 0 % | promedio N=5 | 5,00 | 0,77 | 12,5 | 81,8 |

**Detector B ajustado** (fijo = 0,5; cal. = umbral calibrado por hablante):

| Config. | Estrategia | Umbral | t1 (s) | F1 | FA (%) | Det (%) |
|---|---|---|---|---|---|---|
| 1 s / 0 % | independiente | fijo | 1,0 | 0,79 | 75,0 | 100 |
| 1 s / 0 % | sostenido k=3 | cal. | 4,5 | 0,86 | 16,7 | 81,8 |
| 1 s / 50 % | sostenido k=3 | fijo | 2,4 | 0,80 | 62,5 | 93,9 |
| 1 s / 50 % | sostenido k=3 | cal. | 2,4 | 0,88 | 33,3 | 93,9 |
| 2 s / 0 % | sostenido k=3 | cal. | 7,1 | 0,89 | 16,7 | 69,7 |
| 2 s / 50 % | sostenido k=3 | cal. | 4,9 | 0,85 | 25,0 | 87,9 |
| 4 s / 50 % | sostenido k=3 | cal. | 11,0 | 0,87 | 20,8 | 63,6 |

Lectura:
- Acumular ventanas reduce las falsas alarmas a cambio de decidir más tarde.
- Con umbral fijo, el B ajustado tiene entre 42 y 92 % de falsas alarmas con ventanas de 1 s; la
  calibración las baja a 17 a 67 %, y menos con 2 y 4 s.
- El F1 mayor del B (hasta 0,89) viene de un *recall* más alto: como 33 de 57 flujos son sintéticos,
  decidir siempre "falso" ya da F1 = 0,73. Con la misma FA de 16,7 %, el A detecta 87,9 % a los 2,25 s y
  el B calibrado 81,8 % a los 4,5 s o 69,7 % a los 7,1 s.
- Con 24 flujos reales, un flujo equivale a unos 4 puntos de FA y a 3 de *recall*; diferencias de uno o
  dos flujos son ruido.

## Latencia y RTF (50 % de solape)

| Ventana | A CPU (ms) | B CPU (ms) | B GPU (ms) | RTF A CPU | RTF B CPU | RTF B GPU |
|---|---|---|---|---|---|---|
| 0,5 s | 0,6 | 139,3 | 18,8 | 0,002 | 0,557 | 0,075 |
| 1 s | 0,7 | 153,4 | 18,8 | 0,001 | 0,307 | 0,038 |
| 2 s | 0,8 | 159,7 | 18,6 | 0,001 | 0,160 | 0,019 |
| 3 s | 1,1 | 157,4 | 18,6 | 0,001 | 0,105 | 0,012 |
| 5 s | 1,4 | 157,6 | 18,6 | 0,001 | 0,063 | 0,007 |

GPU: GTX 1650. Máximos observados: 1 196 MB de RAM y 192 MB de GPU. AASIST-L: 85 306 parámetros.

## Prueba con micrófono

Ventana de 2 s, salto de 1 s, `sostenido_k2`, Detector A.
- Voz real del autor: puntaje máximo de 0,575 y ninguna decisión "falso" en 28 ventanas decididas.
- Sintético reproducido en bucle por parlantes (30 s; el audio sonó desde el segundo 5,8 aprox., según
  estimación auditiva): 14 de 28 decisiones "falso" con el detector entrenado con 35 audios y 18 de 28 al
  reprocesar el archivo grabado con el entrenado con los 57 (`results/vivo_reproceso.csv`). Las cuatro
  primeras decisiones "falso" (segundos 3 a 6) corresponden a ventanas con casi solo silencio; entre los
  segundos 7 y 16 todas fueron "real"; desde el 17 hasta el final todas fueron "falso".
- Es una demostración cualitativa con un solo hablante, no una medición de exactitud. El sistema no tiene
  detección de actividad de voz.

## Limitaciones

1. Son 11 hablantes y un solo generador (XTTS v2); no se mide la generalización a otros generadores.
2. El canal sigue confundido con la clase (reales de iPhone, sintéticos de XTTS). Excluir a los dos
   hablantes con AAC no cambió el EER del A de forma apreciable, pero no permite separar el efecto del
   canal.
3. Los sintéticos de spk001 a spk004 usan textos genéricos.
4. Con 24 flujos reales la resolución de las métricas por estrategia es de unos 4 puntos porcentuales.
5. El detector clasifica silencio y ruido de fondo como sintético; falta detección de actividad de voz.
6. La prueba en vivo usa un solo hablante y un canal distinto al de entrenamiento.

## Pendiente para el entregable 4

- Pasar ambas clases por la misma cadena de grabación y compresión.
- Incorporar otros generadores de voz.
- Detección de actividad de voz antes de puntuar.
- Evaluar la fusión de puntajes A+B con los puntajes guardados (`results/puntajes_B_*s.json`).