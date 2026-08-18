"""
Módulo de Procesamiento de Video con Detección YOLO

Este módulo implementa el procesamiento en tiempo real de videos,
ejecutando detección frame por frame y enviando los
resultados como imágenes codificadas en base64.

Componentes principales:
- VideoProcessor: Clase que gestiona la lectura, detección y streaming de video
- Procesamiento multihilo para no bloquear la aplicación principal

"""

import threading
import time
import base64
from queue import Empty, Full, Queue
from typing import Callable, Optional

import cv2
import torch

from core.gpu_manager import current_gpu_memory, require_rtx_5050

# =============================================================================
# GUIA RAPIDA DEL ARCHIVO
# -----------------------------------------------------------------------------

# Arquitectura multihilo:
# - Hilo supervisor (`_run`): crea colas
# - Hilo lector (`_reader_worker`): abre/lee/cierra OpenCV y controla velocidad.
# - Hilo detector (`_detector_worker`): ejecuta CombinedDetector
# - Hilo codificador (`_encoder_worker`): convierte frames anotados a base64.
#
# Por que se separa asi:
# - La lectura de video, la inferencia y la codificacion tienen ritmos distintos
# - La UI queda libre solo recibe callbacks y actualiza controles Flet
#
# Ciclo de vida de modelos:
# - `preload_models()` carga YOLO en segundo plano al iniciar o al reintentar
# - `stop()` detiene el video, pero NO libera modelos
# - `reset_session()` limpia la sesión conservando pesos y buffers en VRAM.
# - `release_models()` libera modelos y memoria GPU solo al cerrar
# =============================================================================

import os
from core.detector_combinado import CombinedDetector


def get_gpu_info() -> dict:
    #Obtiene información completa de la GPU
    info = dict(require_rtx_5050())
    info["memory_used_gb"] = torch.cuda.memory_allocated(
        info["device_index"]
    ) / (1024**3)
    return info


class VideoProcessor:
    """
    Procesador de video con detección de vehículos en tiempo real

    Arquitectura:
    - Hilo supervisor: Gestión y control del pipeline
    """

    def __init__(
        self,
        on_frame: Callable[[str, dict], None],
        on_info: Optional[Callable[[str], None]] = None,
        target_fps: Optional[float] = None,
        enable_vehicles: bool = True,
        enable_lanes: bool = True,
        show_diagnostics: bool = False,
    ):

        # === CONFIGURACIÓN DE GPU ===
        self._gpu_info = get_gpu_info()

        # === CONFIGURACIÓN DE CALLBACKS ===

        self.on_frame = on_frame
        self.on_info = on_info if on_info else lambda msg: None

        self._show_diagnostics = show_diagnostics

        # === CONFIGURACIÓN DE FPS Y VELOCIDAD ===
        self.target_fps = target_fps if target_fps else 30.0
        self._playback_speed = 1.0  
        self._speed_lock = threading.Lock()

        # === CONTROL DE HILO ===

        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._lock = threading.Lock()

        # === CONFIGURACIÓN DE DETECTOR ===

        self._detector: Optional[CombinedDetector] = None

        self._enable_vehicles = enable_vehicles
        self._enable_lanes = enable_lanes
        self._detector_lock = threading.Lock()
        self._models_ready = threading.Event()
        self._preload_thread: Optional[threading.Thread] = None

        # === OPTIMIZACIONES DE RENDIMIENTO ===
        self._jpeg_quality = 65
        self._vehicle_detection_interval = 1
        self._lane_detection_interval = 1

        # === Conteo ===
        self._total_detections = 0
        self._frame_count = 0
        self._total_pesados = 0
        self._total_livianos = 0
        self._total_motos = 0

    def _diagnostic(self, message: str) -> None:
        if self._show_diagnostics:
            self.on_info(message)

    def _wait_while_running(self, seconds: float) -> None:
        deadline = time.perf_counter() + max(0.0, seconds)
        while self._running.is_set():
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.05))

    def models_ready(self) -> bool:
        #ndica si los modelos están cargados y pueden reutilizarse
        return self._models_ready.is_set() and self._detector is not None

    def set_detection_enabled(self, enable_vehicles: bool, enable_lanes: bool):
        #Actualiza módulos activos sin descargar modelos precargados
        self._enable_vehicles = enable_vehicles
        self._enable_lanes = enable_lanes
        with self._detector_lock:
            if self._detector:
                self._detector.enable_vehicles = enable_vehicles
                self._detector.enable_lanes = enable_lanes

    def set_playback_speed(self, speed: float):

        with self._speed_lock:
            self._playback_speed = max(0.25, min(4.0, speed))

    def get_playback_speed(self) -> float:
        #Retorna la velocidad de reproducción actual
        with self._speed_lock:
            return self._playback_speed

    def increase_speed(self) -> float:
        #Control para controlar la velocidad de reproducción
        speeds = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
        current = self.get_playback_speed()
        for s in speeds:
            if s > current:
                self.set_playback_speed(s)
                return s
        return current

    def decrease_speed(self) -> float:
        #Disminuye la velocidad de reproducción
        speeds = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
        current = self.get_playback_speed()
        for s in reversed(speeds):
            if s < current:
                self.set_playback_speed(s)
                return s
        return current

    def start(self, file_path: str) -> bool:
        
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False

        # Activar flag de "corriendo"
        # Esta bandera es consultada por todos los hilos del pipeline. Al limpiarla
        # se les pide terminar de forma ordenada.
        self._running.set()

        # Reiniciar estadísticas
        self._total_detections = 0
        self._frame_count = 0
        self._total_pesados = 0
        self._total_livianos = 0
        self._total_motos = 0

        with self._lock:
            self._thread = threading.Thread(
                target=self._run, args=(file_path,), daemon=True
            )
            self._thread.start()

        # Notificar inicio
        self.on_info(f"Reproduciendo: {os.path.basename(file_path)}")

        # Mostrar info de GPU
        self._diagnostic(f"GPU CUDA dedicada: {self._gpu_info['device_name']}")
        return True

    def stop(self, timeout: float = 8.0):
        """Detiene el procesamiento sin descargar los modelos YOLO."""
        # Parar es intencionalmente barato: corta el pipeline de video, pero deja
        # los modelos en memoria para que el siguiente Inicio sea rapido.
        with self._lock:
            if self._thread and self._thread.is_alive():
                self._running.clear()
                self._thread.join(timeout=timeout)
                if self._thread.is_alive():
                    return False

        self._thread = None

        return True

    def release_models(self, timeout: float = 30.0) -> bool:
        #Libera modelos y memoria GPU. Usar durante el cierre definitivo
        stopped = self.stop(timeout=timeout)
        if not stopped:
            self.on_info("⚠ Hilo detector tarda demasiado en cerrar")
            return False

        with self._detector_lock:
            if self._detector:
                self._detector.reset()
                self._detector = None
            self._models_ready.clear()

        torch.cuda.empty_cache()

        return True

    def reset_session(self, timeout: float = 8.0) -> bool:

        if not self.stop(timeout=timeout):
            return False

        with self._detector_lock:
            if self._detector:

                self._detector.reset(release_cuda_cache=False)

        self._total_detections = 0
        self._frame_count = 0
        self._total_pesados = 0
        self._total_livianos = 0
        self._total_motos = 0
        self.set_playback_speed(1.0)
        return True

    def _report_gpu_memory(self, context: str):

        if not self._show_diagnostics:
            return
        memory = current_gpu_memory()
        self.on_info(
            f"GPU memoria {context}: usada {memory['allocated_gb']:.2f} GB, "
            f"reservada {memory['reserved_gb']:.2f} GB / "
            f"{memory['total_gb']:.1f} GB"
        )

    def _prepare_inference_device(self):
        #Prepara el dispositivo CUDA dentro del hilo de inferencia

        self._gpu_info = get_gpu_info()
        torch.cuda.set_device(self._gpu_info["device_index"])
        self._diagnostic(
            f"Inferencia YOLO fijada en {self._gpu_info['device']}: "
            f"{self._gpu_info['device_name']}"
        )

    def _ensure_detector_loaded(self) -> bool:

        if self._models_ready.is_set() and self._detector is not None:
            return True

        with self._detector_lock:
            if self._models_ready.is_set() and self._detector is not None:
                return True

            try:
                self._prepare_inference_device()
                self._diagnostic("📦 Cargando detectores...")
                self._detector = CombinedDetector(
                    enable_vehicles=self._enable_vehicles,
                    enable_lanes=self._enable_lanes,
                    device=self._gpu_info["device"],
                    vehicle_interval=self._vehicle_detection_interval,
                    lane_interval=self._lane_detection_interval,
                )

                self._models_ready.set()
                self.on_info("✅ Detectores precargados en GPU CUDA dedicada")
                self._diagnostic(
                    f"Inferencia optimizada: vehiculos cada "
                    f"{self._vehicle_detection_interval} frames, carriles cada "
                    f"{self._lane_detection_interval} frames"
                )
                self._report_gpu_memory("tras cargar modelos")
                return True
            except Exception as ex:
                self._models_ready.clear()
                self.on_info(f"❌ Error cargando detectores: {ex}")
                return False

    def preload_models(self, on_done: Optional[Callable[[bool], None]] = None):

        if self._models_ready.is_set():
            if on_done:
                on_done(True)
            return

        if self._preload_thread and self._preload_thread.is_alive():
            return

        def worker():
            self._diagnostic("Precarga de modelos iniciada")
            ok = self._ensure_detector_loaded()
            if on_done:
                on_done(ok)

        self._preload_thread = threading.Thread(target=worker, daemon=True)
        self._preload_thread.start()

    def _put_latest(self, target_queue: Queue, item):

        while self._running.is_set() or item is None:
            try:
                target_queue.put_nowait(item)
                return
            except Full:
                try:
                    target_queue.get_nowait()
                except Empty:
                    pass

        return

    def _reader_worker(self, file_path: str, frame_queue: Queue):

        cap = None
        self._diagnostic("Hilo lector iniciado")
        try:

            while self._running.is_set() and not self._models_ready.is_set():
                time.sleep(0.02)

            if not self._running.is_set():
                return

            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                self.on_info("❌ No se pudo abrir el video.")
                self._running.clear()
                return

            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if src_fps <= 1e-3:
                src_fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            playback_fps = min(self.target_fps, src_fps)
            base_delay = 1.0 / playback_fps
            self._diagnostic(
                f"🎬 Video: {total_frames} frames @ {src_fps:.0f} FPS"
            )

            while self._running.is_set():
                t0 = time.perf_counter()
                current_speed = self.get_playback_speed()
                delay = base_delay / current_speed
                skips = max(0, int(current_speed) - 1)

                ok, frame = cap.read()
                if not ok:
                    break

                self._put_latest(frame_queue, frame)

                for _ in range(skips):
                    if not self._running.is_set():
                        break
                    cap.grab()

                elapsed = time.perf_counter() - t0
                rem = delay - elapsed
                if rem > 0:
                    self._wait_while_running(rem)
        finally:
            # El mismo worker que construye el decoder también lo libera.
            if cap is not None:
                cap.release()
            self._put_latest(frame_queue, None)
            if self._running.is_set():
                self._diagnostic("Video terminado: fin de archivo")
            else:
                self._diagnostic("Video detenido por usuario")
            self._diagnostic("Hilo lector detenido")

    def _detector_worker(self, frame_queue: Queue, result_queue: Queue):

        self._diagnostic("Hilo detector iniciado")
        try:
            if not self._ensure_detector_loaded():
                self._running.clear()
                return

            while self._running.is_set():
                try:
                    frame = frame_queue.get(timeout=0.1)
                except Empty:
                    continue

                if frame is None:
                    break

                current_counts = {"pesados": 0, "livianos": 0, "motos": 0, "lanes": {}}
                frame_to_send = frame

                try:
                    annotated, combined_info = self._detector.detect(frame)
                    frame_to_send = annotated

                    current_counts = combined_info.get("vehicles", current_counts)
                    current_counts["lanes"] = combined_info.get("lanes", {})

                    if combined_info.get("vehicles_fresh", True):
                        self._total_pesados += current_counts["pesados"]
                        self._total_livianos += current_counts["livianos"]
                        self._total_motos += current_counts["motos"]
                        self._total_detections += sum(
                            [
                                current_counts["pesados"],
                                current_counts["livianos"],
                                current_counts["motos"],
                            ]
                        )
                    self._frame_count += 1
                except Exception:
                    pass

                self._put_latest(result_queue, (frame_to_send, current_counts))
        finally:
            self._put_latest(result_queue, None)
            self._diagnostic("Hilo detector detenido")

    def _encoder_worker(self, result_queue: Queue):
        #Codifica frames procesados y los envía a la UI

        self._diagnostic("Hilo codificador iniciado")
        while self._running.is_set():
            try:
                item = result_queue.get(timeout=0.1)
            except Empty:
                continue

            if item is None:
                break

            frame_to_send, current_counts = item
            ok, buf = cv2.imencode(
                ".jpg",
                frame_to_send,
                [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
            )
            if ok and self._running.is_set():
                b64 = base64.b64encode(buf).decode("ascii")
                self.on_frame(b64, current_counts)

        self._diagnostic("Hilo codificador detenido")

    def _run(self, file_path: str):
        #Loop principal de procesamiento con GPU

        frame_queue = Queue(maxsize=1)
        result_queue = Queue(maxsize=1)
        reader_thread = threading.Thread(
            target=self._reader_worker,
            args=(file_path, frame_queue),
            daemon=True,
        )
        detector_thread = threading.Thread(
            target=self._detector_worker,
            args=(frame_queue, result_queue),
            daemon=True,
        )
        encoder_thread = threading.Thread(
            target=self._encoder_worker,
            args=(result_queue,),
            daemon=True,
        )

        try:
            self._diagnostic("Pipeline multihilo iniciando")
            self._report_gpu_memory("al iniciar pipeline")
            # Se arrancan los tres trabajadores despues de crear recursos comunes.
            reader_thread.start()
            detector_thread.start()
            encoder_thread.start()
            self._diagnostic("Pipeline multihilo activo")

            while self._running.is_set() and (
                reader_thread.is_alive()
                or detector_thread.is_alive()
                or encoder_thread.is_alive()
            ):
                time.sleep(0.05)

        finally:
            self._running.clear()
            for worker in (reader_thread, detector_thread, encoder_thread):
                if worker.is_alive():
                    worker.join(timeout=6.0)
                if worker is detector_thread and worker.is_alive():
                    self.on_info("⚠ Hilo detector tarda demasiado en cerrar")
            self._diagnostic("Pipeline multihilo detenido")
            self._report_gpu_memory("al detener pipeline")

            if self._frame_count > 0:
                self.on_info(
                    f"📊 Resumen: {self._total_detections} vehículos en {self._frame_count} frames"
                )
                self.on_info(
                    f"🚛 Pesados: {self._total_pesados} | 🚗 Livianos: {self._total_livianos} | 🏍️ Motos: {self._total_motos}"
                )
            self.on_info("⏹ Video finalizado")
