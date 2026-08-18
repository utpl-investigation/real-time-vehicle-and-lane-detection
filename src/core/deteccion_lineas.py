import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from core.gpu_manager import RequiredGpuError, require_rtx_5050

# =============================================================================
# GUIA RAPIDA DEL ARCHIVO
# -----------------------------------------------------------------------------
# Este modulo detecta lineas de carril usando un modelo YOLO de segmentacion.
# A diferencia del detector de vehiculos, aqui el modelo devuelve mascaras

# Flujo por frame:
# 1. Calcular dos ROI: uno para el carril izquierdo y otro para el derecho.
# 2. Crear un recorte rectangular ampliado que cubre ambos ROI para dar contexto.
# 3. Ejecutar YOLO sobre ese recorte.


class LaneStatus(Enum):
    CENTERED = "CENTRADO"
    DEVIATION_LEFT = "BUSCANDO CARRIL"
    DEVIATION_RIGHT = "BUSCANDO CARRIL"
    SEARCHING = "BUSCANDO CARRIL"


@dataclass
class LaneInfo:
    # Esta estructura agrupa todo lo que las capas superiores necesitan saber
    # sobre el carril: lineas, centro, desviacion y estado.
    left_line: Optional[np.ndarray] = None
    right_line: Optional[np.ndarray] = None
    lane_center_x: Optional[float] = None
    vehicle_center_x: Optional[float] = None
    deviation_px: float = 0.0
    status: LaneStatus = LaneStatus.SEARCHING

    @property
    def left_detected(self) -> bool:
        return self.left_line is not None and len(self.left_line) > 0

    @property
    def right_detected(self) -> bool:
        return self.right_line is not None and len(self.right_line) > 0

    @property
    def both_detected(self) -> bool:
        return self.left_detected and self.right_detected


def configure_gpu() -> dict:
    # Mantener una única fuente de verdad impide que líneas y vehículos terminen
    # accidentalmente en dispositivos diferentes.
    return dict(require_rtx_5050())


class LaneDetector:

    #Características:
    #- Ajuste de líneas con polyfit grado 1 para líneas sólidas
    #- Extrapolación desde horizonte hasta base de imagen
    #- Sistema de alertas de centrado
    #- Overlay visual del carril detectado
  

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        imgsz: int = 512,
        conf: float = 0.50,
        iou: float = 0.45,
        compile_mode: str | bool = False,
    ):

        # === CONFIGURACIÓN DE GPU ===
        self._gpu_info = configure_gpu()

        # === ROI PARA CARRILES (DOS ZONAS: IZQUIERDA Y DERECHA) ===
        self.roi_top_ratio = 0.65
        self.roi_bottom_ratio = 0.85
        self.center_margin = 0.05
        self.roi_width = 0.35

        # === PARÁMETROS DE LÍNEAS Y ALERTAS ===
        self.horizon_ratio = 0.55  # Horizonte más abajo para no extrapolar tanto
        self.deviation_threshold = 50  # Píxeles de umbral para alerta

        # === COLORES DE VISUALIZACIÓN ===
        self.roi_color_left = (255, 200, 0)  # Cyan para ROI izquierdo
        self.roi_color_right = (0, 200, 255)  # Amarillo para ROI derecho
        self.line_color_left = (255, 255, 0)  # Cyan para línea izquierda
        self.line_color_right = (0, 255, 255)  # Amarillo para línea derecha
        self.lane_overlay_color = (0, 255, 0)  # Verde para overlay del carril
        self.lane_band_thickness = 25  # Grosor de la banda de carril
        self.overlay_alpha = 0.35  # Transparencia del overlay
        self._roi_cache = {}

        # === CARGA DEL MODELO ===
        if model_path is None:
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            # El modelo small equilibra segmentación estable y latencia para MVP.
            model_path = os.path.join(base_dir, "models", "best-lineas.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo de líneas no encontrado en: {model_path}\n"
                f"Asegúrese de que 'best-lineas.pt' esté en la carpeta 'models/'"
            )

        self.model = YOLO(model_path)

        #si el dispositivo no es la RTX 5050 elegida,
        # se detiene la carga para que la configuración pueda corregirse.
        selected_device = self._gpu_info["device"]
        if device is not None and str(device) != selected_device:
            raise RequiredGpuError(
                f"Líneas recibió {device!r}; el único dispositivo permitido es "
                f"{selected_device} ({self._gpu_info['device_name']})."
            )
        self.device = selected_device
        torch.cuda.set_device(self._gpu_info["device_index"])

        # === MOVER MODELO A GPU Y OPTIMIZAR ===
        self.model.to(self.device)
        if hasattr(self.model, "fuse"):
            self.model.fuse()

        # === PRECISIÓN FP16 ===
        self.quantize = "fp16"
        self.compile_mode = compile_mode

        # === PARÁMETROS DE INFERENCIA ===
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self._predict_args = {
            "verbose": False,
            "device": self.device,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "quantize": self.quantize,
            "compile": self.compile_mode,
            "rect": False,
            "max_det": 20,
            "retina_masks": False,
            "channels_last": True,
        }

        self._warmup()

    def _warmup(self):
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            with torch.inference_mode():
                self.model.predict(source=dummy, **self._predict_args)
            torch.cuda.synchronize(self._gpu_info["device_index"])
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo preparar YOLO Segmentación en la RTX 5050: {exc}"
            ) from exc

    def _get_left_roi_vertices(self, height: int, width: int) -> np.ndarray:
        # El ROI izquierdo termina cerca del centro de imagen.
        center_x = width // 2
        top_y = int(height * self.roi_top_ratio)
        bottom_y = int(height * self.roi_bottom_ratio)

        # ROI izquierdo: desde el borde izquierdo hasta cerca del centro
        # Parte superior (más estrecha, cerca del horizonte)
        top_right_x = int(center_x - width * self.center_margin)
        top_left_x = int(center_x - width * (self.center_margin + self.roi_width * 0.5))

        #Parte inferior más ancha, cerca del vehículo
        bottom_right_x = int(center_x - width * self.center_margin)
        bottom_left_x = int(width * 0.05) 

        return np.array(
            [
                [
                    [bottom_left_x, bottom_y],
                    [top_left_x, top_y],
                    [top_right_x, top_y],
                    [bottom_right_x, bottom_y],
                ]
            ],
            dtype=np.int32,
        )

    def _get_right_roi_vertices(self, height: int, width: int) -> np.ndarray:
        # El ROI derecho empieza cerca del centro y se extiende al borde derecho
        center_x = width // 2
        top_y = int(height * self.roi_top_ratio)
        bottom_y = int(height * self.roi_bottom_ratio)

        #ROI derecho: desde cerca del centro hasta el borde derecho
        #Parte superior más estrecha, cerca del horizonte
        top_left_x = int(center_x + width * self.center_margin)
        top_right_x = int(
            center_x + width * (self.center_margin + self.roi_width * 0.5)
        )

        # Parte inferior más ancha, cerca del vehículo
        bottom_left_x = int(center_x + width * self.center_margin)
        bottom_right_x = int(width * 0.95) 

        return np.array(
            [
                [
                    [bottom_left_x, bottom_y],
                    [top_left_x, top_y],
                    [top_right_x, top_y],
                    [bottom_right_x, bottom_y],
                ]
            ],
            dtype=np.int32,
        )

    def _get_expanded_bbox_from_vertices(
        self, vertices: np.ndarray, height: int, width: int, margin_ratio: float = 0.20
    ) -> Tuple[int, int, int, int]:
        # YOLO necesita un recorte rectangular. Este metodo toma los vertices del
        # trapecio y genera un bounding box ampliado para conservar contexto.
        points = vertices.reshape(-1, 2)
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

    def _merge_bboxes(
        self, *bboxes: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        # Como hay dos ROI, se unen ambos rectangulos para hacer una sola inferencia
        # sobre un recorte que cubra izquierda y derecha.
        x1 = min(bbox[0] for bbox in bboxes)
        y1 = min(bbox[1] for bbox in bboxes)
        x2 = max(bbox[2] for bbox in bboxes)
        y2 = max(bbox[3] for bbox in bboxes)
        return x1, y1, x2, y2

    def _get_cached_roi_data(self, height: int, width: int) -> dict:
        cache_key = (height, width)
        if cache_key not in self._roi_cache:
            left_vertices = self._get_left_roi_vertices(height, width)
            right_vertices = self._get_right_roi_vertices(height, width)
            left_crop_bbox = self._get_expanded_bbox_from_vertices(
                left_vertices,
                height,
                width,
            )
            right_crop_bbox = self._get_expanded_bbox_from_vertices(
                right_vertices,
                height,
                width,
            )
            crop_bbox = self._merge_bboxes(left_crop_bbox, right_crop_bbox)
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox

            self._roi_cache[cache_key] = {
                "left_vertices": left_vertices,
                "right_vertices": right_vertices,
                "left_roi_mask": self._create_left_roi_mask(height, width),
                "right_roi_mask": self._create_right_roi_mask(height, width),
                "crop_bbox": crop_bbox,
                "crop_width": crop_x2 - crop_x1,
                "crop_height": crop_y2 - crop_y1,
            }
        return self._roi_cache[cache_key]

    def _create_left_roi_mask(self, height: int, width: int) -> np.ndarray:
        # Mascara binaria: 255 dentro del ROI izquierdo, 0 fuera.
        mask = np.zeros((height, width), dtype=np.uint8)
        vertices = self._get_left_roi_vertices(height, width)
        cv2.fillPoly(mask, vertices, 255)
        return mask

    def _create_right_roi_mask(self, height: int, width: int) -> np.ndarray:
        #Misma idea que el ROI izquierdo, pero para la zona derecha del carril.
        mask = np.zeros((height, width), dtype=np.uint8)
        vertices = self._get_right_roi_vertices(height, width)
        cv2.fillPoly(mask, vertices, 255)
        return mask

    def _draw_roi(self, frame: np.ndarray) -> np.ndarray:
        # Los ROI se dibujan siempre para que el conductor vea que zona analiza el sistema.
        height, width = frame.shape[:2]
        roi_data = self._get_cached_roi_data(height, width)

        # Dibujar ROI izquierdo
        cv2.polylines(frame, roi_data["left_vertices"], True, self.roi_color_left, 2)

        # Dibujar ROI derecho
        cv2.polylines(frame, roi_data["right_vertices"], True, self.roi_color_right, 2)

        return frame

    def _select_aligned_contours(
        self, contours: List[np.ndarray], width: int
    ) -> List[np.ndarray]:
        if not contours:
            return []

        image_center = width / 2.0
        ordered = sorted(
            contours,
            key=lambda contour: abs(float(np.mean(contour[:, 0])) - image_center),
        )
        reference_center = float(np.mean(ordered[0][:, 0]))
        alignment_threshold = width * 0.05
        return [
            contour
            for contour in ordered
            if abs(float(np.mean(contour[:, 0])) - reference_center)
            <= alignment_threshold
        ]

    def _rasterize_contours(
        self, contours: List[np.ndarray], height: int, width: int
    ) -> Optional[np.ndarray]:
        if not contours:
            return None
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [contour.astype(np.int32) for contour in contours], 255)
        return mask

    def _extract_centerline_from_mask(self, mask: np.ndarray) -> Optional[np.ndarray]:
        # Extrae todos los pixeles positivos y los agrupa por coordenada Y para
        #formar una linea central limpia de la mascara segmentada.
        if mask is None or not np.any(mask > 0):
            return None

        ys, xs = np.nonzero(mask > 0)
        if len(ys) < 2:
            return None

        order = np.argsort(ys)
        ys = ys[order]
        xs = xs[order]

        unique_y, start_indices, counts = np.unique(
            ys,
            return_index=True,
            return_counts=True,
        )
        if len(unique_y) < 2:
            return None

        sum_x = np.add.reduceat(xs, start_indices)
        center_x = (sum_x / counts).astype(np.int32)

        return np.column_stack((center_x, unique_y.astype(np.int32)))

    def _apply_roi_to_mask(
        self, segmentation_mask: np.ndarray, roi_mask: np.ndarray
    ) -> np.ndarray:
        #Aplica el ROI a la máscara de segmentación.
        if segmentation_mask.shape[:2] != roi_mask.shape[:2]:
            segmentation_mask = cv2.resize(
                segmentation_mask.astype(np.uint8),
                (roi_mask.shape[1], roi_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return cv2.bitwise_and(segmentation_mask, roi_mask)

    def _collect_all_points_from_masks(
        self,
        masks_data: List[np.ndarray],
        roi_mask: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:

        all_points = []

        for mask in masks_data:
            # Aplicar ROI
            mask_filtered = self._apply_roi_to_mask(mask, roi_mask)

            if np.any(mask_filtered > 0):
                # Extraer centerline de esta máscara
                centerline = self._extract_centerline_from_mask(mask_filtered)
                if centerline is not None and len(centerline) > 0:
                    all_points.extend(centerline.tolist())

        if len(all_points) == 0:
            return np.array([])

        return np.array(all_points, dtype=np.int32)

    def _calculate_lane_center(
        self,
        left_line: Optional[np.ndarray],
        right_line: Optional[np.ndarray],
        height: int,
    ) -> Optional[float]:
        # El centro del carril se calcula promediando la posicion X de la linea izquierda y derecha cerca de la parte baja del frame.

        if left_line is None or right_line is None:
            return None

        if len(left_line) == 0 or len(right_line) == 0:
            return None

        # Obtener X de cada línea en la parte inferior (Y máximo)
        left_bottom = left_line[left_line[:, 1].argmax()]
        right_bottom = right_line[right_line[:, 1].argmax()]

        left_x = left_bottom[0]
        right_x = right_bottom[0]

        return (left_x + right_x) / 2.0

    def _determine_lane_status(
        self, lane_center_x: Optional[float], vehicle_center_x: float
    ) -> Tuple[LaneStatus, float]:
        # Compara centro del vehiculo/camara contra centro del carril y decide si esta centrado o si debe seguir buscando referencias confiables.
        if lane_center_x is None:
            return LaneStatus.SEARCHING, 0.0

        deviation = vehicle_center_x - lane_center_x

        if abs(deviation) <= self.deviation_threshold:
            return LaneStatus.CENTERED, deviation
        elif deviation > 0:
            return LaneStatus.DEVIATION_RIGHT, deviation
        else:
            return LaneStatus.DEVIATION_LEFT, deviation

    def _draw_lane_overlay(
        self, frame: np.ndarray, left_line: np.ndarray, right_line: np.ndarray
    ) -> np.ndarray:

        if left_line is None or right_line is None:
            return frame

        if len(left_line) < 2 or len(right_line) < 2:
            return frame

        try:
            # Crear polígono combinando ambas líneas
            # Ordenar left_line de arriba a abajo y right_line de abajo a arriba
            left_sorted = left_line[left_line[:, 1].argsort()]
            right_sorted = right_line[right_line[:, 1].argsort()][::-1]

            # Combinar para formar el polígono del carril
            polygon = np.vstack([left_sorted, right_sorted])

            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], self.lane_overlay_color)

            # Mezclar con el frame original
            frame = cv2.addWeighted(
                overlay, self.overlay_alpha, frame, 1 - self.overlay_alpha, 0
            )

        except Exception as e:
            pass

        return frame

    def _draw_lane_band(
        self,
        frame: np.ndarray,
        line_points: np.ndarray,
        color: Tuple[int, int, int],
        alpha: float = 0.6,
    ) -> np.ndarray:

        if line_points is None or len(line_points) < 2:
            return frame

        # Crear overlay
        overlay = frame.copy()

        # Dibujar línea gruesa como banda
        cv2.polylines(
            overlay,
            [line_points],
            isClosed=False,
            color=color,
            thickness=self.lane_band_thickness,
        )

        # Mezclar con transparencia
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Dibujar borde más oscuro encima
        border_color = tuple(int(c * 0.6) for c in color)
        cv2.polylines(
            frame, [line_points], isClosed=False, color=border_color, thickness=3
        )

        return frame

    def _smooth_line_polyfit(
        self, points: np.ndarray, height: int, width: int, degree: int = 2
    ) -> Optional[np.ndarray]:

        if points is None or len(points) < 3:
            return points if points is not None and len(points) >= 2 else None

        try:
            x = points[:, 0].astype(np.float64)
            y = points[:, 1].astype(np.float64)

            # Ajustar polinomio: x = f(y)
            coeffs = np.polyfit(y, x, degree)

            # Usar el rango Y de los puntos detectados (sin extrapolar)
            y_min = max(y.min(), int(height * self.horizon_ratio))
            y_max = min(y.max(), int(height * self.roi_bottom_ratio))

            # Generar puntos suavizados
            y_smooth = np.linspace(y_min, y_max, 80)
            x_smooth = np.polyval(coeffs, y_smooth)

            # Filtrar puntos dentro del frame
            valid_mask = (x_smooth >= 0) & (x_smooth < width)
            y_smooth = y_smooth[valid_mask]
            x_smooth = x_smooth[valid_mask]

            if len(x_smooth) < 2:
                return points

            return np.column_stack(
                (x_smooth.astype(np.int32), y_smooth.astype(np.int32))
            )

        except Exception:
            return points

    def _draw_status_indicator(
        self, frame: np.ndarray, lane_info: LaneInfo
    ) -> np.ndarray:

        height, width = frame.shape[:2]

        # Configurar colores según estado
        if lane_info.status == LaneStatus.CENTERED:
            color = (0, 255, 0)  # Verde
            bg_color = (0, 100, 0)
        else:
            color = (0, 255, 255)  # Amarillo
            bg_color = (0, 100, 100)

        # Texto del estado
        text = lane_info.status.value
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # Calcular tamaño del texto
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Posición: parte superior central
        x = (width - text_width) // 2
        y = 40

        # Dibujar fondo
        padding = 10
        cv2.rectangle(
            frame,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            bg_color,
            -1,
        )
        cv2.rectangle(
            frame,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            color,
            2,
        )

        # Dibujar texto
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

        # Mostrar desviación si hay carril detectado
        if lane_info.both_detected and lane_info.deviation_px != 0:
            direction = "derecha" if lane_info.deviation_px > 0 else "izquierda"
            dev_text = f"Desviacion: {abs(lane_info.deviation_px):.0f}px {direction}"
            cv2.putText(frame, dev_text, (x, y + 30), font, 0.5, (255, 255, 255), 1)

        # Dibujar indicador visual del centro
        if lane_info.both_detected and lane_info.lane_center_x is not None:
            # Línea del centro del carril
            lane_cx = int(lane_info.lane_center_x)
            cv2.line(
                frame, (lane_cx, height - 100), (lane_cx, height - 20), (0, 255, 0), 2
            )

            # Marcador del centro del vehículo
            vehicle_cx = int(lane_info.vehicle_center_x)
            cv2.line(
                frame,
                (vehicle_cx, height - 100),
                (vehicle_cx, height - 20),
                (255, 255, 255),
                2,
            )

        return frame

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:

        height, width = frame.shape[:2]
        vehicle_center_x = width / 2.0

        roi_data = self._get_cached_roi_data(height, width)
        left_roi_mask = roi_data["left_roi_mask"]
        right_roi_mask = roi_data["right_roi_mask"]
        crop_x1, crop_y1, crop_x2, crop_y2 = roi_data["crop_bbox"]
        inference_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # Ejecutar inferencia en GPU
        with torch.inference_mode():
            results = self.model.predict(
                source=inference_frame,
                **self._predict_args,
            )

        result = results[0]
        annotated = frame

        # Mantener contornos hasta elegir los mejores evita crear una máscara de
        #tamaño completo por cada instancia devuelta por YOLO
        left_contours = []
        right_contours = []
        total_masks = 0

        try:
            mask_contours = result.masks.xy if result.masks is not None else []
            if mask_contours:
                for contour in mask_contours:
                    total_masks += 1
                    if contour is None or len(contour) < 3:
                        continue
                    contour_adjusted = contour.copy()
                    contour_adjusted[:, 0] += crop_x1
                    contour_adjusted[:, 1] += crop_y1
                    if float(np.mean(contour_adjusted[:, 0])) < vehicle_center_x:
                        left_contours.append(contour_adjusted)
                    else:
                        right_contours.append(contour_adjusted)

        except Exception:
            pass

        # Rasterizar como máximo una máscara combinada por lado
        best_left_mask = self._rasterize_contours(
            self._select_aligned_contours(left_contours, width), height, width
        )
        best_right_mask = self._rasterize_contours(
            self._select_aligned_contours(right_contours, width), height, width
        )

        # Recolectar puntos solo de la mejor máscara de cada lado
        left_points = np.array([])
        right_points = np.array([])

        if best_left_mask is not None:
            left_points = self._collect_all_points_from_masks(
                [best_left_mask], left_roi_mask, height, width
            )

        if best_right_mask is not None:
            right_points = self._collect_all_points_from_masks(
                [best_right_mask], right_roi_mask, height, width
            )

        left_line = None
        right_line = None

        if len(left_points) >= 3:
            left_line = self._smooth_line_polyfit(left_points, height, width, degree=2)
        elif len(left_points) >= 2:
            left_line = left_points

        if len(right_points) >= 3:
            right_line = self._smooth_line_polyfit(
                right_points, height, width, degree=2
            )
        elif len(right_points) >= 2:
            right_line = right_points

        # Crear info del carril
        lane_info = LaneInfo(
            left_line=left_line,
            right_line=right_line,
            vehicle_center_x=vehicle_center_x,
        )

        # Calcular centro del carril si ambas líneas están detectadas
        if lane_info.both_detected:
            lane_info.lane_center_x = self._calculate_lane_center(
                left_line, right_line, height
            )

        # Determinar estado de centrado
        lane_info.status, lane_info.deviation_px = self._determine_lane_status(
            lane_info.lane_center_x, vehicle_center_x
        )

        # === VISUALIZACIÓN ===

        # 1. Dibujar overlay verde del carril (primero, para que quede debajo)
        if lane_info.both_detected:
            annotated = self._draw_lane_overlay(annotated, left_line, right_line)

        # 2. Dibujar bandas de carril (líneas gruesas con transparencia)
        if left_line is not None and len(left_line) >= 2:
            annotated = self._draw_lane_band(
                annotated, left_line, self.line_color_left, alpha=0.7
            )

        if right_line is not None and len(right_line) >= 2:
            annotated = self._draw_lane_band(
                annotated, right_line, self.line_color_right, alpha=0.7
            )

        # 3. Dibujar indicador de estado
        annotated = self._draw_status_indicator(annotated, lane_info)

        # Preparar diccionario de info para compatibilidad
        info = {
            "total_masks": total_masks,
            "masks_in_left_roi": len(left_contours),
            "masks_in_right_roi": len(right_contours),
            "left_detected": lane_info.left_detected,
            "right_detected": lane_info.right_detected,
            "both_detected": lane_info.both_detected,
            "lane_center_x": lane_info.lane_center_x,
            "vehicle_center_x": lane_info.vehicle_center_x,
            "deviation_px": lane_info.deviation_px,
            "status": lane_info.status.value,
            "detected_lines_left": 1 if lane_info.left_detected else 0,
            "detected_lines_right": 1 if lane_info.right_detected else 0,
        }

        return annotated, info

    def reset(self, release_cuda_cache: bool = True):
        """Reinicia caches geométricas y, opcionalmente, el allocator CUDA."""
        self._roi_cache.clear()
        if release_cuda_cache:
            torch.cuda.empty_cache()
