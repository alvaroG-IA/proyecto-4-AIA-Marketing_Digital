# 🚀 Proyecto: Agente de IA para Generación de Entornos de Producto

![Status](https://img.shields.io/badge/Status-Funcional-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/Framework-LangGraph-red)
![IA](https://img.shields.io/badge/Enfoque-Generative%20AI-orange)

## 📋 Descripción del Proyecto
Este proyecto implementa un **Agente Inteligente de Orquestación** basado en grafos para la edición avanzada de fotografía de E-commerce. El sistema permite transformar una imagen de producto básica (ej. una botella sobre un fondo neutro) en una imagen publicitaria.

### Fases y Componentes Principales
1. **Director de Arte (Prompt Optimization):** Un nodo basado en **Llama 3.2** transforma una idea simple del usuario en prompts técnicos optimizados para modelos de difusión, enfocándose en iluminación de estudio y texturas realistas.
2. **Segmentación Inteligente (SAM):** Utiliza el **Segment Anything Model (SAM)** de Meta para identificar y aislar el objeto principal de la imagen de forma automática, generando una máscara precisa mediante la detección del punto central.
3. **Enrutamiento Adaptativo:** El agente detecta automáticamente si existe conexión a internet para decidir el motor de renderizado:
    * **Modo Online:** Utiliza la API de **Replicate** para ejecutar el modelo **FLUX.1 Fill**, obteniendo resultados de máxima calidad en la nube.
    * **Modo Offline:** Utiliza una pipeline local de **Stable Diffusion + ControlNet (Canny)** mediante la librería `diffusers` para garantizar la privacidad y el funcionamiento sin red.
4. **Arquitectura de Grafo (LangGraph):** Toda la lógica se lleva a cabo por un grafo de estados que gestiona el flujo de datos entre nodos y la persistencia de la información.

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python
* **Orquestación:** `langgraph`, `langchain`
* **Modelos de Lenguaje:** `Ollama` (Llama 3.2)
* **Segmentación:** `segment-anything` (SAM)
* **Generación de Imagen:** `replicate` (API), `diffusers` (Local), `ControlNet`
* **Procesamiento de Imagen:** `opencv-python`, `Pillow`, `numpy`

## 📋 Características Principales
* **Optimización Estructurada:** Uso de Pydantic para garantizar que el LLM devuelva prompts técnicos específicos para diferentes modelos (SD y Flux).
* **Procesamiento de Máscaras:** Aplicación de dilatación y desenfoque gaussiano para una integración natural (blending) del producto en el nuevo entorno.
* **Resiliencia de Red:** Capacidad de conmutar entre inferencia local y en la nube de forma transparente para el usuario.
* **Control de Composición:** Implementación de ControlNet Canny en modo local para preservar la silueta y detalles del producto original durante el proceso de inpainting.

## 📂 Estructura del Repositorio (Directorio `src/`)
```bash
├── agent/
│   ├── nodes/
│   │   ├── node_llm.py          # Nodo: Optimizador de Prompts (Llama 3.2)
│   │   ├── node_sam.py          # Nodo: Segmentador (SAM)
│   │   ├── node_api.py          # Nodo: Generador en la Nube (Flux @ Replicate)
│   │   └── node_local.py        # Nodo: Generador Local (SD + ControlNet)
│   ├── graph.py                 # Definición de la arquitectura LangGraph
│   ├── state.py                 # Definición del esquema de estado del agente
│   ├── router.py                # Lógica de decisión (Internet Check)
│   └── main.py                  # Punto de entrada de la aplicación
├── data/
│   └── images/                  # Imágenes de entrada del usuario
└── output/                      # Resultados, máscaras y procesados
```

## 🚀 Instalación y Entorno
1. Crear un entorno
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

2. instalar dependencias
```bash
pip install -r requirements.txt
```

3. Configuración de Variables de Entorno:
Dentro de .venv meter el Token para poder usar la API
```bash
REPLICATE_API_TOKEN=tu_token_aqui
```

4. Modelos Locales:
- Asegúrate de tener Ollama instalado y el modelo descargado: ollama run llama3.2.
- El script descargará automáticamente el checkpoint de SAM (sam_vit_b_01ec64.pth) en la primera ejecución.

# 🎛️ Guía de Uso del Sistema Principal

Para ejecutar el agente y transformar una imagen, sigue estos pasos:

- Preparar la imagen: Coloca tu imagen de producto en data/images/botella.png (o modifica la ruta en main.py).
- Ejecutar el Agente:
```bash
python src/agent/main.py
```
- Interacción: El sistema te pedirá una descripción del entorno deseado.

Al finalizar, el sistema mostrará la ruta de la imagen final en la carpeta output/, detallando si el proceso fue Local o Online según tu disponibilidad de conexión.

