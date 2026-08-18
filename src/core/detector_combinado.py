"""
Módulo de Detección Combinada: Vehículos + Líneas de Carril
Combina la detección de vehículos y la detección de líneas de carril con YOLO 
ara GPU NVIDIA con CUDA.
"""

import numpy as np
import cv2
import torch
from typing import Tuple, Dict

from core.gpu_manager import RequiredGpuError, require_rtx_5050

from core.deteccion_vehiculos import VehicleDetector
from core.deteccion_lineas import LaneDetector


class CombinedDetector:

    def __init__(
        self,
        enable_vehicles: bool = True,
        enable_lanes: bool = True,
        device: str | None = None,
        vehicle_interval: int = 1,
        lane_interval: int = 1,
    ):

        """
        Args:
            enable_vehicles: Habilitar detección de vehículos
            enable_lanes: Habilitar detección de carriles
            device: Índice CUDA seleccionado para la RTX 5050; None lo detecta
        """
        # Checks que permiten activar/desactivar cada detector desde la UI
        self.enable_vehicles = enable_vehicles
        self.enable_lanes = enable_lanes
        self.vehicle_interval = max(1, int(vehicle_interval))
        self.lane_interval = max(1, int(lane_interval))

        self._frame_index = 0

        self._last_vehicle_counts = {"pesados": 0, "livianos": 0, "motos": 0}
        self._last_lane_info = {
            "left_detected": False,
            "right_detected": False,
            "both_detected": False,
            "status": "Buscando carril...",
            "deviation_px": 0,
            "lane_center_x": None,
            "vehicle_center_x": None,
        }

        # Todos los módulos comparten la selección estricta de la RTX 5050
        self._gpu_info = self._get_gpu_info()
        selected_device = self._gpu_info["device"]
        if device is not None and str(device) != selected_device:
            raise RequiredGpuError(
                f"Detector combinado recibió {device!r}; se requiere "
                f"{selected_device} ({self._gpu_info['device_name']})."
            )
        self.device = selected_device

        # Los detectores se crean solo si estan habilitados
        self.vehicle_detector = None
        self.lane_detector = None

        if enable_vehicles:
            self.vehicle_detector = VehicleDetector(device=self.device)

        if enable_lanes:
            self.lane_detector = LaneDetector(device=self.device)

    def _get_gpu_info(self) -> dict:
        #Obtiene información de la GPU
        info = dict(require_rtx_5050())
        info["memory_total"] = info["memory_total_gb"]
        return info

    def _draw_center_line(self, frame: np.ndarray) -> np.ndarray:
        #Dibuja una línea vertical blanca fina en el centro del frame.
        height, width = frame.shape[:2]
        center_x = width // 2
        cv2.line(frame, (center_x, 0), (center_x, height), (255, 255, 255), 1)
        return frame

    def _draw_fixed_rois(self, frame: np.ndarray) -> np.ndarray:
        #Dibuja los ROI de carriles y vehículos en todos los frames
        if self.enable_lanes and self.lane_detector:
            frame = self.lane_detector._draw_roi(frame)
        if self.enable_vehicles and self.vehicle_detector:
            frame = self.vehicle_detector._draw_roi(frame)
        return frame

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        #Ejecuta detección combinada usando GPU
        self._frame_index += 1

        annotated = frame
        # Los intervalos permiten saltar inferencias para mejorar rendimiento. Con
        # valor 1 se ejecutan en todos los frames.
        run_lanes = (
            self._frame_index == 1 or self._frame_index % self.lane_interval == 0
        )
        run_vehicles = (
            self._frame_index == 1 or self._frame_index % self.vehicle_interval == 0
        )

        combined_info = {
            "vehicles": dict(self._last_vehicle_counts),
            "lanes": dict(self._last_lane_info),
            "vehicles_fresh": False,
            "lanes_fresh": False,
        }

        # 1. Detección de líneas 
        # Primero carriles: el overlay queda como fondo visual
        if self.enable_lanes and self.lane_detector and run_lanes:
            annotated, lane_info = self.lane_detector.detect(annotated)
            self._last_lane_info = lane_info
            combined_info["lanes"] = lane_info
            combined_info["lanes_fresh"] = True

        # 2. Detección de vehículos 
        # Luego vehiculos: las cajas quedan encima del overlay de carril
        if self.enable_vehicles and self.vehicle_detector and run_vehicles:
            annotated, vehicle_result = self.vehicle_detector.detect(annotated)
            if hasattr(vehicle_result, "_roi_counts"):
                self._last_vehicle_counts = vehicle_result._roi_counts
                combined_info["vehicles"] = vehicle_result._roi_counts
                combined_info["vehicles_fresh"] = True

        # 3. Dibujar línea central blanca
        if not combined_info["lanes"].get("both_detected", False):
            annotated = self._draw_center_line(annotated)

        annotated = self._draw_fixed_rois(annotated)

        return annotated, combined_info

    def reset(self, release_cuda_cache: bool = True):
        #Reinicia estado y, opcionalmente, libera reservas del allocator CUDA

        self._frame_index = 0
        self._last_vehicle_counts = {"pesados": 0, "livianos": 0, "motos": 0}
        self._last_lane_info = {
            "left_detected": False,
            "right_detected": False,
            "both_detected": False,
            "status": "Buscando carril...",
            "deviation_px": 0,
            "lane_center_x": None,
            "vehicle_center_x": None,
        }
        if self.lane_detector:
            self.lane_detector.reset(release_cuda_cache=release_cuda_cache)
        if self.vehicle_detector:
            self.vehicle_detector.reset(release_cuda_cache=release_cuda_cache)

        if release_cuda_cache:
            torch.cuda.empty_cache()
