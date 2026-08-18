# Real-Time Vehicle and Lane Detection in Urban Environments Using YOLO11m

### **Luis Barba-Guaman** y **Jordy Corrales Zapata**

Universidad Técnica Particular de Loja, Departamento de Ciencias de Computación y Electrónica, Ecuador  
✉️ <lrbarba@utpl.edu.ec>
✉️ <jacorrales@utpl.edu.ec>

---

## Etapa de recolección de datos 
A continuación se presentan los Project Id para acceder a las dos fuentes de datos, tanto para las detección de vehículos **car-xg2un** como a las líneas de carril **line-rtzim**, para poder visualizar el contenido el usuario debe estar registrado y haber iniciado sesión en la plataforma Roboflow, solicitar acceso mediante los link [Vehículos](https://app.roboflow.com/join/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3Jrc3BhY2VJZCI6Ilg4NzVRNlhXZllodHp0aGptNDh1MHU4dm1kdjIiLCJyb2xlIjoib3duZXIiLCJpbnZpdGVyIjoiamFjb3JyYWxlc0B1dHBsLmVkdS5lYyIsImlhdCI6MTc4NjkwNjcxOX0.igzbcIkix5d5HKlkdDljPO9FUxe-7IpxJGFcmW92j5Q) y [Líneas de carril](https://app.roboflow.com/join/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3Jrc3BhY2VJZCI6Ilg4NzVRNlhXZllodHp0aGptNDh1MHU4dm1kdjIiLCJyb2xlIjoib3duZXIiLCJpbnZpdGVyIjoiamFjb3JyYWxlc0B1dHBsLmVkdS5lYyIsImlhdCI6MTc4NjkwNjgwM30.8P15N6FVCFCzGlboCriu-eDYM9lRwNizN7BSthULHWs), estas imágenes tienen sus respectivas anotaciones y clases.

---

## Etapa de entrenamiento en Google Colab

Para la etapa de entrenamiento se usó Google Colab y los archivos de ejecución tanto para la detección de vehículos y detección de carriles, se encuentran en la carpeta :open_file_folder: `/training`

Estos archivos requieren una API de Roboflow para acceder a las fuentes de datos

Los modelos entrenados se encuentran en la carpeta :open_file_folder: `/models`

---

## Etapa de desarrollo del prototipo

El prototipo fue desarrollado y ejecutado en una laptop Acer con las siguientes características:
- AMD Ryzen 5 240
- 16 GB de RAM
- NVIDIA GeForce RTX 5050
- Windows 11 

El entorno de desarrollo se ejecutó en
- Python 3.14.5
- Flet

---

## Descripción del prototipo

El código se encuentran en la carpeta :open_file_folder: `/src`

#### Interfaz de Usuario (`src/ui/`)

  * Interfaz gráfica interactiva desarrollada con **Flet**.
  * Administra la visualización del video en tiempo real, controles de reproducción (play, pausa, velocidad), activar/desactivar detectores y paneles de métricas como conteo de vehículos, desviación del carril, FPS, estado de GPU y alertas.

#### Detección y Procesamiento (`src/core/`)

* **video_processor.py**
  * Motor de procesamiento de video multi-hilo.
  * Implementa un pipeline asíncrono con tres hilos independientes (*Lector*, *Detector* y *Codificador*)

* **detector_combinado.py**
  * Coordina la detección vehicular y el análisis de líneas de carril sobre cada fotograma.
  * Dibuja los elementos visuales en el orden adecuado carriles de fondo, vehículos al frente y consolida las métricas.

* **deteccion_vehiculos.py**
  * Módulo de detección de vehículos mediante YOLO OBB (Oriented Bounding Boxes)
  * Delimita una Región de Interés (ROI) trapezoidal y clasifica/cuenta vehículos en tres categorías: Pesados, Livianos y Motos.

* **deteccion_lineas.py**
  * Módulo de detección y segmentación de carriles mediante YOLO Segmentation.
  * Extrae contornos de líneas, ajusta curvas mediante polinomios, extrapola la trayectoria, calcula la desviación respecto al centro del vehículo y genera el overlay visual.

* **gpu_manager.py**
  * Gestor y validador de hardware para GPU NVIDIA RTX con CUDA.


