import os
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from core.gpu_manager import RequiredGpuError, require_rtx_5050

# =============================================================================
# GUIA RAPIDA DEL ARCHIVO
# -----------------------------------------------------------------------------
# Este modulo se encarga solo de vehiculos. Recibe frames OpenCV, ejecuta YOLO
# OBB (cajas orientadas) sobre una region de interes y devuelve el mismo frame
# con dibujos mas conteos por categoria.
#
# Flujo por frame:
# 1. Calcular o reutilizar el ROI segun la resolucion del video.
# 2. Ejecutar inferencia FP16 en la RTX 5050 con `torch.inference_mode()`.
# 3 Convertir las cajas desde coordenadas del recorte al frame completo.
# 4. Dibujar cajas orientadas sobre el frame.
#
# =============================================================================

def configure_gpu() -> dict:
    # La selección centralizada evita que cada detector elija un índice distinto.
    info = dict(require_rtx_5050())
    info["memory_total"] = info["memory_total_gb"]
    return info


class VehicleDetector:

    CLASES_MODELO = {
        "VehiculoPesado": "pesado",
        "VehiculoLiviano": "liviano",
        "Moto": "moto",
    }

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        imgsz: int = 512,
        conf: float = 0.70,
        iou: float = 0.45,
        compile_mode: str | bool = False,
    ):
 
        # === CONFIGURACIÓN DE GPU ===
        self._gpu_info = configure_gpu()

        # === CARGA DEL MODELO ===
        if model_path is None:
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )

            model_path = os.path.join(base_dir, "models", "best-vehiculos.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo no encontrado en: {model_path}\n"
                f"Asegúrese de que 'best-vehiculos.pt' esté en la carpeta 'models/'"
            )

        self.model = YOLO(model_path)

        # Se rechaza cualquier petición de CPU y se verifica que el índice recibido sea el seleccionado para la RTX 5050.
        selected_device = self._gpu_info["device"]
        if device is not None and str(device) != selected_device:
            raise RequiredGpuError(
                f"Vehículos recibió {device!r}; el único dispositivo permitido "
                f"es {selected_device} ({self._gpu_info['device_name']})."
            )
        self.device = selected_device
        torch.cuda.set_device(self._gpu_info["device_index"])

        # === MOVER MODELO A GPU Y OPTIMIZAR ===
        self.model.to(self.device)
        if hasattr(self.model, "fuse"):
            self.model.fuse()

        self.quantize = "fp16"

        self.compile_mode = compile_mode

        # === PARÁMETROS DE INFERENCIA ===
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        # Limitar detecciones y fijar forma evita trabajo innecesario de NMS y
        #mantiene una ruta eficiente incluso sin depender de TorchInductor
        self._predict_args = {
            "verbose": False,
            "device": self.device,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "quantize": self.quantize,
            "compile": self.compile_mode,
            "rect": False,
            "max_det": 50,
            "channels_last": True,
        }

        self._model_class_names = {}
        self._roi_cache = {}

        try:
            self._model_class_names = dict(self.model.names)
        except Exception:
            pass

        self._warmup()

    def _warmup(self):
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            with torch.inference_mode():
                self.model.predict(source=dummy, **self._predict_args)
            torch.cuda.synchronize(self._gpu_info["device_index"])
        except Exception as e:
            raise RuntimeError(f"No se pudo preparar YOLO OBB en la RTX 5050: {e}") from e

    def _get_roi_polygon(self, height: int, width: int) -> np.ndarray:
        # El ROI es trapezoidal para ajustarse a la perspectiva de una camara
        roi_top_y = int(height * 0.55)
        top_left_x = int(width * 0.35)
        top_right_x = int(width * 0.65)
        roi_bottom_y = int(height * 0.85)
        bottom_left_x = int(width * 0.05)
        bottom_right_x = int(width * 0.95)

        return np.array(
            [
                [bottom_left_x, roi_bottom_y],
                [top_left_x, roi_top_y],
                [top_right_x, roi_top_y],
                [bottom_right_x, roi_bottom_y],
            ],
            dtype=np.int32,
        )

    def _get_expanded_bbox_from_points(
        self, points: np.ndarray, height: int, width: int, margin_ratio: float = 0.20
    ) -> Tuple[int, int, int, int]:

        xs = points[:, 0]
        ys = points[:, 1]
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        bbox_w = x2 - x1
        bbox_h = y2 - y1
        margin_x = int(bbox_w * margin_ratio)
        margin_y = int(bbox_h * margin_ratio)

        return (
            max(0, x1 - margin_x),
            max(0, y1 - margin_y),
            min(width, x2 + margin_x),
            min(height, y2 + margin_y),
        )

    def _get_cached_roi_data(self, height: int, width: int) -> dict:
        cache_key = (height, width)
        if cache_key not in self._roi_cache:
            roi_polygon = self._get_roi_polygon(height, width)
            self._roi_cache[cache_key] = {
                "roi_polygon": roi_polygon,
                "crop_bbox": self._get_expanded_bbox_from_points(
                    roi_polygon,
                    height,
                    width,
                ),
            }
        return self._roi_cache[cache_key]

    def _tensor_to_numpy(self, value: Any) -> np.ndarray:
        #Convierte tensores de Ultralytics a NumPy

        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.array(value)

    def _iter_obb_detections(self, result, offset_x: int, offset_y: int):

        obb = getattr(result, "obb", None)
        if obb is None or len(obb) == 0:
            return

        # Una sola transferencia GPU→CPU evita dos sincronizaciones por frame.
        packed = torch.cat(
            (obb.xyxyxyxy.reshape(-1, 8), obb.cls.reshape(-1, 1)), dim=1
        )
        packed_array = self._tensor_to_numpy(packed)

        for detection in packed_array:
            points = detection[:8].reshape(4, 2)
            cls_idx = int(detection[8])
            box = {
                "x1": float(points[0][0]) + offset_x,
                "y1": float(points[0][1]) + offset_y,
                "x2": float(points[1][0]) + offset_x,
                "y2": float(points[1][1]) + offset_y,
                "x3": float(points[2][0]) + offset_x,
                "y3": float(points[2][1]) + offset_y,
                "x4": float(points[3][0]) + offset_x,
                "y4": float(points[3][1]) + offset_y,
            }
            cls_name = self._model_class_names.get(cls_idx, str(cls_idx))
            yield cls_name, box

    def _is_point_in_roi(self, x: float, y: float, roi_polygon: np.ndarray) -> bool:
        #Verifica si un punto está dentro del ROI
        result = cv2.pointPolygonTest(roi_polygon, (float(x), float(y)), False)
        return result >= 0

    def _is_detection_in_roi(self, box: dict, roi_polygon: np.ndarray) -> bool:
        #Verifica si una detección está dentro del ROI.
        x_center = (
            box.get("x1", 0) + box.get("x2", 0) + box.get("x3", 0) + box.get("x4", 0)
        ) / 4
        y_center = (
            box.get("y1", 0) + box.get("y2", 0) + box.get("y3", 0) + box.get("y4", 0)
        ) / 4
        return self._is_point_in_roi(x_center, y_center, roi_polygon)

    def _draw_obb_box(
        self,
        frame: np.ndarray,
        box: dict,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        #Dibuja una caja OBB en el frame
        points = np.array(
            [
                [int(box.get("x1", 0)), int(box.get("y1", 0))],
                [int(box.get("x2", 0)), int(box.get("y2", 0))],
                [int(box.get("x3", 0)), int(box.get("y3", 0))],
                [int(box.get("x4", 0)), int(box.get("y4", 0))],
            ],
            dtype=np.int32,
        )

        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=thickness)
        return frame

    def _draw_roi(self, frame: np.ndarray) -> np.ndarray:
        #Dibuja el ROI en el frame
        height, width = frame.shape[:2]
        roi_points = self._get_cached_roi_data(height, width)["roi_polygon"]
        cv2.polylines(
            frame, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2
        )
        return frame

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Any]:
        #Ejecuta detección de vehículos en un frame usando GPU
        height, width = frame.shape[:2]
        roi_data = self._get_cached_roi_data(height, width)
        roi_polygon = roi_data["roi_polygon"]
        crop_x1, crop_y1, crop_x2, crop_y2 = roi_data["crop_bbox"]
        inference_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        with torch.inference_mode():
            results = self.model.predict(
                source=inference_frame,
                **self._predict_args,
            )

        result = results[0]
        annotated = frame

        # Contadores por categoria mostrados en la interfaz.
        pesados = 0
        livianos = 0
        motos = 0

        try:
            # Se recorren las cajas OBB ya corregidas al frame original, se filtran por ROI real y se dibujan con color diferente segun categoria.
            for cls_name, box in self._iter_obb_detections(result, crop_x1, crop_y1):
                if self._is_detection_in_roi(box, roi_polygon):
                    if cls_name == "VehiculoPesado":
                        pesados += 1
                        color = (0, 0, 255)
                    elif cls_name == "VehiculoLiviano":
                        livianos += 1
                        color = (255, 255, 0)
                    elif cls_name == "Moto":
                        motos += 1
                        color = (255, 0, 255)
                    else:
                        color = (255, 255, 255)

                    annotated = self._draw_obb_box(
                        annotated, box, color=color, thickness=2
                    )

        except Exception:
            pass

        # Guardamos conteos en el objeto resultado 
        result._roi_counts = {"pesados": pesados, "livianos": livianos, "motos": motos}

        return annotated, result

    def reset(self, release_cuda_cache: bool = True):
        #Reinicia caches
        self._roi_cache.clear()
        if release_cuda_cache:
            torch.cuda.empty_cache()
