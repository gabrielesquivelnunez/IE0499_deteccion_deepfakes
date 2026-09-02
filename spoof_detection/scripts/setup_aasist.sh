#!/bin/bash
# setup_aasist.sh
#
# Clona el repositorio oficial de AASIST (clovaai/aasist), que incluye
# la arquitectura y los pesos preentrenados sobre ASVspoof2019 LA
# (models/weights/AASIST.pth y AASIST-L.pth). No hace falta descargarlos
# aparte: ya vienen dentro del repo.
set -e

cd "$(dirname "$0")/.."  # raíz de spoof_detection/
mkdir -p external
cd external

if [ -d "aasist" ]; then
    echo "external/aasist ya existe, no se vuelve a clonar."
else
    git clone --depth 1 https://github.com/clovaai/aasist.git
fi

echo ""
echo "Listo. Pesos preentrenados disponibles en:"
ls -la aasist/models/weights/
echo ""
echo "Siguiente paso: instalar PyTorch (no incluido en requirements.txt por su tamaño):"
echo "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
echo "Luego probar la carga del modelo con:"
echo "  python -m models.detector_b_aasist"
