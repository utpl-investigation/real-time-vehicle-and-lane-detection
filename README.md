# Real-Time Vehicle and Lane Detection in Urban Environments Using YOLO11m

### **Luis Barba-Guaman** y **Jordy Corrales Zapata**

Universidad Técnica Particular de Loja, Departamento de Ciencias de Computación y Electrónica, Ecuador  
✉️ <lrbarba@utpl.edu.ec>
✉️ <jacorrales@utpl.edu.ec>

---

## Etapa de recolección de datos 
A continuación se presentan los Project Id para acceder a las dos fuentes de datos, tanto para la detección de vehículos **car-xg2un** como a las líneas de carril **line-rtzim**, para poder visualizar el contenido el usuario debe estar registrado e iniciado sección en la plataforma Roboflow, solicitar acceso mediante los link [Vehículos](https://app.roboflow.com/join/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3Jrc3BhY2VJZCI6Ilg4NzVRNlhXZllodHp0aGptNDh1MHU4dm1kdjIiLCJyb2xlIjoib3duZXIiLCJpbnZpdGVyIjoiamFjb3JyYWxlc0B1dHBsLmVkdS5lYyIsImlhdCI6MTc4NjkwNjcxOX0.igzbcIkix5d5HKlkdDljPO9FUxe-7IpxJGFcmW92j5Q) y [Líneas de carril](https://app.roboflow.com/join/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3Jrc3BhY2VJZCI6Ilg4NzVRNlhXZllodHp0aGptNDh1MHU4dm1kdjIiLCJyb2xlIjoib3duZXIiLCJpbnZpdGVyIjoiamFjb3JyYWxlc0B1dHBsLmVkdS5lYyIsImlhdCI6MTc4NjkwNjgwM30.8P15N6FVCFCzGlboCriu-eDYM9lRwNizN7BSthULHWs), estas imágenes tienen sus respectivas anotaciones y clases.

---

## Etapa de entrenamiento en Google Colab

Para la etapa de entrenamiento se uso Google Colab y los archivo de ejecución tanto para la detección de vehículos y detección de carriles, se encuentran en la carpeta :open_file_folder: `/training`

Estos archivos requieren una API de roboflow para acceder a las fuentes de datos

---
