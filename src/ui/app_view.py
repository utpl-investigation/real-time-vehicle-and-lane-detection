import atexit
import asyncio
import threading
import time
from collections import deque

import flet as ft

COLORS = ft.Colors
ICONS = ft.Icons

from core.video_processor import VideoProcessor


def main(page: ft.Page):
    """Función principal de la aplicación Flet."""

    # === CONFIGURACIÓN DE LA PÁGINA ===
    page.title = "Detección de Líneas de Carril y Vehículos"
    page.padding = 0
    page.bgcolor = "#0B1622"
    page.window.maximized = True
    page.window.resizable = True

    # === PALETA DE COLORES ===
    TOPBAR_BG = "#0C1A29"
    PANEL_BG = "#0D1B2A"
    SIDEBAR_BG = "#E3E3E3"
    BTN_BLUE = "#2E6BE5"
    BTN_GREEN = "#2EAD3B"
    BTN_RED = "#E53935"
    BTN_ORANGE = "#F39C12"
    BTN_GRAY = "#607D8B"

    detection_state = {
        "enable_vehicles": True,
        "enable_lanes": True,
        "is_playing": False,  
        "models_ready": False,
        "is_preloading": False,
    }

    app_state = {"closing": False}

    ui_inbox_lock = threading.Lock()
    pending_frame = {"payload": None}
    pending_messages = deque(maxlen=50)
    pending_ui_actions = deque(maxlen=50)

    def enqueue_ui_action(action):

        if app_state["closing"]:
            return
        with ui_inbox_lock:
            pending_ui_actions.append(action)

    def discard_pending_frame():
        #Evita mostrar un frame tardío después de Parar, Limpiar o Subir
        with ui_inbox_lock:
            pending_frame["payload"] = None

    # === HELPER: CREAR BOTONES ===
    def btn(texto, color, on_click, icon=None):
        # Helper para que todos los botones principales compartan estilo y solo cambien texto, color, icono y accion.
        return ft.Button(
            content=texto,
            icon=icon,
            on_click=on_click,
            style=ft.ButtonStyle(
                color=COLORS.WHITE,
                bgcolor=color,
                shape=ft.RoundedRectangleBorder(radius=24),
                padding=ft.Padding.symmetric(horizontal=22, vertical=12),
            ),
        )

    # === COMPONENTES DE UI ===
    # Lista de mensajes operativos: carga de modelos, hilos activos, fin de video, advertencias de GPU y errores
    alerts = ft.ListView(expand=1, spacing=6, padding=10)

    # Contadores de vehículos
    count_pesados = ft.Text("0", size=24, weight=ft.FontWeight.BOLD, color=COLORS.WHITE)
    count_livianos = ft.Text(
        "0", size=24, weight=ft.FontWeight.BOLD, color=COLORS.WHITE
    )
    count_motos = ft.Text("0", size=24, weight=ft.FontWeight.BOLD, color=COLORS.WHITE)

    # Indicadores de estado del carril (izquierda y derecha)
    lane_left_icon = ft.Icon(ICONS.CHEVRON_LEFT, size=20, color=COLORS.ORANGE)
    lane_left_status_text = ft.Text(
        "Buscando...",
        size=11,
        weight=ft.FontWeight.BOLD,
        color=COLORS.ORANGE,
        text_align=ft.TextAlign.CENTER,
    )

    lane_right_icon = ft.Icon(ICONS.CHEVRON_RIGHT, size=20, color=COLORS.ORANGE)
    lane_right_status_text = ft.Text(
        "Buscando...",
        size=11,
        weight=ft.FontWeight.BOLD,
        color=COLORS.ORANGE,
        text_align=ft.TextAlign.CENTER,
    )

    lane_deviation_text = ft.Text(
        "", size=10, color=COLORS.BLACK_54, text_align=ft.TextAlign.CENTER
    )

    lane_left_container = ft.Container(
        content=ft.Column(
            [
                lane_left_icon,
                ft.Text("Izquierda", size=9, color=COLORS.BLACK_54),
                lane_left_status_text,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=2,
        ),
        bgcolor="#FFF3E0",
        border_radius=10,
        padding=ft.Padding.all(8),
        border=ft.Border.all(2, COLORS.ORANGE),
        expand=True,
    )

    lane_right_container = ft.Container(
        content=ft.Column(
            [
                lane_right_icon,
                ft.Text("Derecha", size=9, color=COLORS.BLACK_54),
                lane_right_status_text,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=2,
        ),
        bgcolor="#FFF3E0",
        border_radius=10,
        padding=ft.Padding.all(8),
        border=ft.Border.all(2, COLORS.ORANGE),
        expand=True,
    )

    lane_status_row = ft.Row([lane_left_container, lane_right_container], spacing=8)

    def create_counter_card(icon, label, count_text, bg_color):
        # Crea una tarjeta compacta para cada categoria de vehiculo. Recibe el
        # control `count_text` para actualizarlo sin reconstruir la UI.
        return ft.Container(
            content=ft.Column(
                [
                    ft.Icon(icon, size=28, color=COLORS.WHITE),
                    count_text,
                    ft.Text(label, size=10, color=COLORS.WHITE_70),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=2,
            ),
            bgcolor=bg_color,
            border_radius=12,
            padding=ft.Padding.symmetric(vertical=10, horizontal=15),
            expand=True,
        )

    counters_row = ft.Row(
        [
            create_counter_card(
                ICONS.LOCAL_SHIPPING, "Pesados", count_pesados, "#C62828"
            ),
            create_counter_card(
                ICONS.DIRECTIONS_CAR, "Livianos", count_livianos, "#1565C0"
            ),
            create_counter_card(ICONS.TWO_WHEELER, "Motos", count_motos, "#E65100"),
        ],
        spacing=8,
    )

    # === SWITCHES DE DETECCIÓN ===
    switch_vehicles = ft.Switch(value=True, active_track_color=COLORS.GREEN)
    switch_lanes = ft.Switch(value=True, active_track_color=COLORS.BLUE)

    # Texto de estado de los switches
    switch_status_text = ft.Text(
        "✓ Opciones disponibles", size=10, color=COLORS.GREEN, italic=True
    )

    def update_switches_state():
        # Los switches se bloquean durante reproduccion o precarga para evitar que
        # el usuario cambie opciones mientras el pipeline esta usando modelos
        is_playing = detection_state["is_playing"]
        models_ready = detection_state["models_ready"]
        is_preloading = detection_state["is_preloading"]
        switch_vehicles.disabled = is_playing or is_preloading
        switch_lanes.disabled = is_playing or is_preloading

        if is_playing:
            switch_status_text.value = "⚠ Pause para cambiar"
            switch_status_text.color = COLORS.ORANGE
        elif is_preloading:
            switch_status_text.value = "Precargando modelos..."
            switch_status_text.color = COLORS.ORANGE
        elif not models_ready:
            # Tras un fallo se habilitan los switches para que el usuario pueda
            # reintentar con uno de los detectores desactivado
            switch_status_text.value = "Precarga fallida; puede reintentar"
            switch_status_text.color = COLORS.RED
        else:
            switch_status_text.value = "✓ Opciones disponibles"
            switch_status_text.color = COLORS.GREEN

    def on_switch_vehicles(e):
        # Cambia la deteccion de vehiculos sin reconstruir la UI. Si el procesador
        # ya existe, tambien actualiza sus banderas internas
        if detection_state["is_playing"]:
            return  # No permitir cambios si está reproduciendo
        detection_state["enable_vehicles"] = switch_vehicles.value
        if processor_ref["processor"]:
            processor_ref["processor"].set_detection_enabled(
                detection_state["enable_vehicles"], detection_state["enable_lanes"]
            )
        estado = "activada" if switch_vehicles.value else "desactivada"
        alert_row = ft.Row(
            [
                ft.Icon(
                    ICONS.DIRECTIONS_CAR,
                    size=16,
                    color=COLORS.GREEN if switch_vehicles.value else COLORS.GREY,
                ),
                ft.Text(
                    f"Detección de vehículos {estado}", color=COLORS.BLACK_87, size=12
                ),
            ],
            spacing=6,
        )
        alerts.controls.insert(0, alert_row)
        if len(alerts.controls) > 50:
            alerts.controls.pop()
        page.update()

    def on_switch_lanes(e):
        # Cambia la deteccion de carriles sin recargar modelos; solo actualiza la
        # configuracion que usara el procesador.
        if detection_state["is_playing"]:
            return  # No permitir cambios si está reproduciendo
        detection_state["enable_lanes"] = switch_lanes.value
        if processor_ref["processor"]:
            processor_ref["processor"].set_detection_enabled(
                detection_state["enable_vehicles"], detection_state["enable_lanes"]
            )
        estado = "activada" if switch_lanes.value else "desactivada"
        alert_row = ft.Row(
            [
                ft.Icon(
                    ICONS.ROUTE,
                    size=16,
                    color=COLORS.BLUE if switch_lanes.value else COLORS.GREY,
                ),
                ft.Text(
                    f"Detección de carriles {estado}", color=COLORS.BLACK_87, size=12
                ),
            ],
            spacing=6,
        )
        alerts.controls.insert(0, alert_row)
        if len(alerts.controls) > 50:
            alerts.controls.pop()
        page.update()

    switch_vehicles.on_change = on_switch_vehicles
    switch_lanes.on_change = on_switch_lanes

    detection_controls = ft.Container(
        content=ft.Column(
            [
                ft.Text(
                    "Opciones de Detección",
                    size=14,
                    weight=ft.FontWeight.BOLD,
                    color=COLORS.BLACK_87,
                ),
                switch_status_text,
                ft.Container(height=4),
                ft.Row(
                    [
                        switch_vehicles,
                        ft.Icon(ICONS.DIRECTIONS_CAR, size=16, color=COLORS.BLACK_54),
                        ft.Text("Vehículos", size=12, color=COLORS.BLACK_87),
                    ],
                    spacing=6,
                ),
                ft.Row(
                    [
                        switch_lanes,
                        ft.Icon(ICONS.ROUTE, size=16, color=COLORS.BLACK_54),
                        ft.Text("Carriles", size=12, color=COLORS.BLACK_87),
                    ],
                    spacing=6,
                ),
            ],
            spacing=4,
        ),
        padding=ft.Padding.all(8),
        bgcolor="#D0D0D0",
        border_radius=8,
    )

    # Placeholder cuando no hay video
    video_placeholder = ft.Column(
        [
            ft.Icon(ICONS.VIDEO_FILE, size=64, color=COLORS.WHITE_54),
            ft.Text("Sin video cargado", color=COLORS.WHITE, size=16, opacity=0.8),
            ft.Text(
                "Presione 'Subir' para cargar un video", color=COLORS.WHITE_54, size=12
            ),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=10,
    )

    # Placeholder de cargando
    loading_placeholder = ft.Column(
        [
            ft.ProgressRing(width=48, height=48, stroke_width=4, color=COLORS.BLUE),
            ft.Text(
                "Cargando...", color=COLORS.WHITE, size=16, weight=ft.FontWeight.BOLD
            ),
            ft.Text("Inicializando detectores", color=COLORS.WHITE_54, size=12),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=10,
    )

    # Vista de imagen
    image_view = ft.Image(
        src="", expand=True, fit=ft.BoxFit.CONTAIN, gapless_playback=True
    )

    # Indicador de estado
    status_indicator = ft.Container(
        content=ft.Row(
            [
                ft.Icon(ICONS.CIRCLE, size=12, color=COLORS.RED),
                ft.Text("Detenido", color=COLORS.WHITE, size=12),
            ],
            spacing=4,
        ),
        padding=ft.Padding.symmetric(horizontal=10, vertical=5),
        border_radius=20,
        bgcolor="#1A1A2E",
    )
    # Se muestran los frames que realmente llegan a
    # Flet, no los FPS declarados por el archivo de origen.
    fps_icon = ft.Icon(ICONS.SPEED, size=16, color=COLORS.GREY)
    fps_value_text = ft.Text(
        "FPS: --", color=COLORS.WHITE, size=12, weight=ft.FontWeight.BOLD
    )
    fps_level_text = ft.Text("", color=COLORS.GREY, size=10)
    fps_indicator = ft.Container(
        content=ft.Row(
            [
                fps_icon,
                ft.Column(
                    [fps_value_text, fps_level_text],
                    spacing=0,
                    tight=True,
                    horizontal_alignment=ft.CrossAxisAlignment.START,
                ),
            ],
            spacing=6,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=ft.Padding.symmetric(horizontal=10, vertical=4),
        border_radius=20,
        bgcolor="#1A1A2E",
        tooltip="FPS de frames procesados y presentados en pantalla",
    )

    # Control de velocidad
    speed_text = ft.Text("1.0x", color=COLORS.WHITE, size=14, weight=ft.FontWeight.BOLD)

    def on_speed_decrease(_):
        # Reduce la velocidad de reproduccion desde el procesador y refleja el
        # nuevo valor en la barra superior
        if processor_ref["processor"]:
            new_speed = processor_ref["processor"].decrease_speed()
            speed_text.value = f"{new_speed}x"
            page.update()

    def on_speed_increase(_):
        # Aumenta la velocidad de reproduccion desde el procesador y actualiza la
        # etiqueta
        if processor_ref["processor"]:
            new_speed = processor_ref["processor"].increase_speed()
            speed_text.value = f"{new_speed}x"
            page.update()

    speed_control = ft.Container(
        content=ft.Row(
            [
                ft.IconButton(
                    icon=ICONS.FAST_REWIND,
                    icon_size=18,
                    icon_color=COLORS.WHITE,
                    tooltip="Más lento",
                    on_click=on_speed_decrease,
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Icon(ICONS.SPEED, size=14, color=COLORS.WHITE_70),
                            speed_text,
                        ],
                        spacing=0,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    width=50,
                ),
                ft.IconButton(
                    icon=ICONS.FAST_FORWARD,
                    icon_size=18,
                    icon_color=COLORS.WHITE,
                    tooltip="Más rápido",
                    on_click=on_speed_increase,
                ),
            ],
            spacing=0,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        padding=ft.Padding.symmetric(horizontal=8, vertical=2),
        border_radius=20,
        bgcolor="#1A1A2E",
    )

    # Cronómetro de tiempo real de la ejecución actual
    processing_time_text = ft.Text(
        "00:00", color=COLORS.WHITE, size=14, weight=ft.FontWeight.BOLD
    )
    processing_timer = ft.Container(
        content=ft.Row(
            [
                ft.Icon(ICONS.TIMER, size=17, color=COLORS.CYAN),
                processing_time_text,
            ],
            spacing=6,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=ft.Padding.symmetric(horizontal=10, vertical=7),
        border_radius=20,
        bgcolor="#1A1A2E",
        tooltip="Tiempo transcurrido de procesamiento del video",
    )
    processing_timer_state = {
        "running": False,
        "started_at": 0.0,
        "elapsed": 0.0,
        "displayed_second": 0,
    }

    def format_processing_time(total_seconds: int) -> str:
        #Devuelve duración MM:SS 
        minutes, seconds = divmod(max(0, total_seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"

    def reset_processing_timer():
        #Prepara el cronómetro para una nueva ejecución de video
        processing_timer_state["running"] = False
        processing_timer_state["started_at"] = 0.0
        processing_timer_state["elapsed"] = 0.0
        processing_timer_state["displayed_second"] = 0
        processing_time_text.value = "00:00"

    def start_processing_timer():
        
        reset_processing_timer()
        processing_timer_state["started_at"] = time.perf_counter()
        processing_timer_state["running"] = True

    def refresh_processing_timer() -> bool:
       
        if not processing_timer_state["running"]:
            return False

        elapsed = time.perf_counter() - processing_timer_state["started_at"]
        processing_timer_state["elapsed"] = elapsed
        current_second = int(elapsed)
        if current_second == processing_timer_state["displayed_second"]:
            return False

        processing_timer_state["displayed_second"] = current_second
        processing_time_text.value = format_processing_time(current_second)
        return True

    def stop_processing_timer():
        #Congela el valor al terminar el video o detener la inferencia
        if not processing_timer_state["running"]:
            return

        elapsed = time.perf_counter() - processing_timer_state["started_at"]
        processing_timer_state["elapsed"] = elapsed
        processing_timer_state["running"] = False
        final_second = int(elapsed)
        processing_timer_state["displayed_second"] = final_second
        processing_time_text.value = format_processing_time(final_second)

    # Panel de video
    video_panel = ft.Container(
        expand=True,
        bgcolor=PANEL_BG,
        border_radius=16,
        alignment=ft.Alignment.CENTER,
        content=video_placeholder,
        padding=10,
    )

    # === REFERENCIA AL PROCESADOR ===
    processor_ref = {"processor": None}

    # Estado para controlar si ya llegó el primer frame
    frame_state = {
        "first_frame_received": False,
        "current_filename": "",
        "last_stats_update": 0.0,
        "fps_window_started": time.perf_counter(),
        "fps_frames": 0,
        "fps_last_frame_at": 0.0,
        "fps_stalled": False,
    }

    def set_fps_indicator(fps: float | None):
        #Clasifica los FPS efectivos y cambia solo los controles del indicador
        if fps is None:
            fps_value_text.value = "FPS: --"
            fps_level_text.value = ""
            color = COLORS.GREY
        elif fps >= 20.0:
            fps_value_text.value = f"FPS: {fps:.1f}"
            fps_level_text.value = "Alto"
            color = COLORS.GREEN
        elif fps >= 15.0:
            fps_value_text.value = f"FPS: {fps:.1f}"
            fps_level_text.value = "Aceptable"
            color = COLORS.AMBER
        elif fps >= 10.0:
            fps_value_text.value = f"FPS: {fps:.1f}"
            fps_level_text.value = "Bajo"
            color = COLORS.ORANGE
        else:
            fps_value_text.value = f"FPS: {fps:.1f}"
            fps_level_text.value = "No aceptable"
            color = COLORS.RED

        fps_icon.color = color
        fps_level_text.color = color

    def reset_fps_metric():
        #Reinicia la ventana de medición al cambiar el estado del video
        frame_state["fps_window_started"] = time.perf_counter()
        frame_state["fps_frames"] = 0
        frame_state["fps_last_frame_at"] = 0.0
        frame_state["fps_stalled"] = False
        set_fps_indicator(None)

    # === RENDERIZADO EN EL EVENT LOOP DE FLET ===
    def _apply_frame_with_counts(b64: str, counts: dict):
        # Esta función solo la invoca `ui_event_pump`,
        if app_state["closing"]:
            return
        try:
            # El primer frame cambia el contenido del panel de "cargando" a imagen.
            first_frame = not frame_state["first_frame_received"]

            # Si es el primer frame, cambiar de loading a image_view
            if first_frame:
                frame_state["first_frame_received"] = True
                video_panel.content = ft.Column(
                    [
                        ft.Container(
                            content=ft.Row(
                                [
                                    ft.Icon(
                                        ICONS.VIDEO_FILE, size=16, color=COLORS.WHITE_54
                                    ),
                                    ft.Text(
                                        frame_state["current_filename"],
                                        color=COLORS.WHITE,
                                        size=12,
                                        overflow=ft.TextOverflow.ELLIPSIS,
                                    ),
                                ],
                                spacing=6,
                            ),
                            padding=ft.Padding.only(bottom=8),
                        ),
                        ft.Container(content=image_view, expand=True),
                    ],
                    expand=True,
                )

            # Flet recibe el mismo texto Base64 mediante 
            image_view.src = b64

            now = time.perf_counter()
            fps_updated = False
            if first_frame or frame_state["fps_stalled"]:
                frame_state["fps_window_started"] = now
                frame_state["fps_frames"] = 1
                frame_state["fps_stalled"] = False
            else:
                frame_state["fps_frames"] += 1
                fps_elapsed = now - frame_state["fps_window_started"]
                if fps_elapsed >= 1.0:
                    effective_fps = frame_state["fps_frames"] / fps_elapsed
                    set_fps_indicator(effective_fps)
                    frame_state["fps_window_started"] = now
                    frame_state["fps_frames"] = 0
                    fps_updated = True
            frame_state["fps_last_frame_at"] = now

            count_pesados.value = str(counts.get("pesados", 0))
            count_livianos.value = str(counts.get("livianos", 0))
            count_motos.value = str(counts.get("motos", 0))

            # Actualizar estado del carril
            lane_info = counts.get("lanes", {})
            lane_status = lane_info.get("status", "🔍 Buscando carril...")
            deviation = lane_info.get("deviation_px", 0)
            both_detected = lane_info.get("both_detected", False)
            left_detected = lane_info.get("left_detected", False)
            right_detected = lane_info.get("right_detected", False)

            # Actualizar indicador IZQUIERDO
            if left_detected:
                lane_left_status_text.value = "✓ Detectado"
                lane_left_status_text.color = "#2E7D32"  # Verde oscuro
                lane_left_icon.color = COLORS.GREEN
                lane_left_container.bgcolor = "#E8F5E9"  # Verde claro
                lane_left_container.border = ft.Border.all(2, COLORS.GREEN)
            else:
                lane_left_status_text.value = "Buscando..."
                lane_left_status_text.color = "#F57C00"  # Naranja
                lane_left_icon.color = COLORS.ORANGE
                lane_left_container.bgcolor = "#FFF3E0"  # Naranja claro
                lane_left_container.border = ft.Border.all(2, COLORS.ORANGE)

            # Actualizar indicador DERECHO
            if right_detected:
                lane_right_status_text.value = "✓ Detectado"
                lane_right_status_text.color = "#2E7D32"  # Verde oscuro
                lane_right_icon.color = COLORS.GREEN
                lane_right_container.bgcolor = "#E8F5E9"  # Verde claro
                lane_right_container.border = ft.Border.all(2, COLORS.GREEN)
            else:
                lane_right_status_text.value = "Buscando..."
                lane_right_status_text.color = "#F57C00"  # Naranja
                lane_right_icon.color = COLORS.ORANGE
                lane_right_container.bgcolor = "#FFF3E0"  # Naranja claro
                lane_right_container.border = ft.Border.all(2, COLORS.ORANGE)


            # Mostrar desviación si está detectado
            if both_detected and deviation != 0:
                direction = "→" if deviation > 0 else "←"
                lane_deviation_text.value = (
                    f"Desviación: {abs(deviation):.0f}px {direction}"
                )
            else:
                lane_deviation_text.value = ""

            if app_state["closing"]:
                return

            if first_frame:
                page.update()
            else:
                # En frames normales se refresca solo la imagen, no toda la pagina
                image_view.update()
                if fps_updated:

                    fps_indicator.update()

                if now - frame_state["last_stats_update"] >= 0.25:
                    frame_state["last_stats_update"] = now
                    count_pesados.update()
                    count_livianos.update()
                    count_motos.update()
                    lane_left_container.update()
                    lane_right_container.update()
                    lane_deviation_text.update()
        except Exception:
            pass

    def _apply_info(msg):
        # Convierte un mensaje ya entregado al event loop en una fila visual
        if app_state["closing"]:
            return
        try:
            if "Detectores precargados" in msg:
                detection_state["models_ready"] = True
                detection_state["is_preloading"] = False
                icon, icon_color = ICONS.CHECK_CIRCLE, COLORS.GREEN
                status_indicator.content.controls[0].color = COLORS.GREEN
                status_indicator.content.controls[1].value = (
                    "Procesando" if detection_state["is_playing"] else "Listo"
                )
                update_switches_state()
            elif "Error cargando detectores" in msg:
                detection_state["models_ready"] = False
                detection_state["is_preloading"] = False
                icon, icon_color = ICONS.ERROR, COLORS.RED
                status_indicator.content.controls[0].color = COLORS.RED
                status_indicator.content.controls[1].value = "Error de modelos"
                update_switches_state()
            elif "✅" in msg or "listo" in msg.lower():
                icon, icon_color = ICONS.CHECK_CIRCLE, COLORS.GREEN
            elif "❌" in msg or "error" in msg.lower():
                icon, icon_color = ICONS.ERROR, COLORS.RED
            elif "⚠" in msg:
                icon, icon_color = ICONS.WARNING, COLORS.ORANGE
            elif "▶" in msg:
                icon, icon_color = ICONS.PLAY_ARROW, COLORS.GREEN
                status_indicator.content.controls[0].color = COLORS.GREEN
                status_indicator.content.controls[1].value = "Procesando"
                # Marcar como reproduciendo y deshabilitar switches
                detection_state["is_playing"] = True
                update_switches_state()
            elif "Precarga de modelos iniciada" in msg:
                icon, icon_color = ICONS.INVENTORY, COLORS.AMBER
                status_indicator.content.controls[0].color = COLORS.ORANGE
                status_indicator.content.controls[1].value = "Precargando"
            elif (
                "Precarga de modelos finalizada" in msg
                or "Detectores precargados" in msg
            ):
                icon, icon_color = ICONS.CHECK_CIRCLE, COLORS.GREEN
                status_indicator.content.controls[0].color = COLORS.RED
                status_indicator.content.controls[1].value = "Detenido"
            elif "⏹" in msg or "finalizado" in msg.lower():
                icon, icon_color = ICONS.STOP, COLORS.BLUE
                stop_processing_timer()
                status_indicator.content.controls[0].color = COLORS.RED
                status_indicator.content.controls[1].value = "Detenido"
                # Marcar como detenido y habilitar switches
                detection_state["is_playing"] = False
                update_switches_state()
            elif "📊" in msg or "📈" in msg:
                icon, icon_color = ICONS.ANALYTICS, COLORS.PURPLE
            elif "🎬" in msg:
                icon, icon_color = ICONS.MOVIE, COLORS.CYAN
            elif "📦" in msg:
                icon, icon_color = ICONS.INVENTORY, COLORS.AMBER
            elif "🚛" in msg or "🚗" in msg or "🏍️" in msg:
                icon, icon_color = ICONS.SUMMARIZE, COLORS.TEAL
            else:
                icon, icon_color = ICONS.INFO, COLORS.BLUE

            clean_text = msg
            for emoji in [
                "✅",
                "❌",
                "⚠",
                "▶",
                "⏹",
                "📊",
                "📈",
                "🎬",
                "📦",
                "🖥️",
                "⚡",
                "🏷️",
                "🚛",
                "🚗",
                "🏍️",
            ]:
                clean_text = clean_text.replace(emoji, "")

            alert_row = ft.Row(
                [
                    ft.Icon(icon, size=16, color=icon_color),
                    ft.Text(
                        clean_text.strip(),
                        color=COLORS.BLACK_87,
                        size=12,
                        expand=True,
                        selectable=True,
                    ),
                ],
                spacing=6,
            )

            alerts.controls.insert(0, alert_row)
            if len(alerts.controls) > 50:
                alerts.controls.pop()
            if app_state["closing"]:
                return
            page.update()
        except Exception:
            pass

    # === CALLBACKS PARA LOS HILOS DEL PROCESADOR ===
    def on_frame_with_counts(b64: str, counts: dict):

        if app_state["closing"]:
            return
        with ui_inbox_lock:

            pending_frame["payload"] = (b64, counts)

    def on_info(msg):

        if app_state["closing"]:
            return
        with ui_inbox_lock:
            pending_messages.append(msg)

    async def ui_event_pump():

        while not app_state["closing"]:
            with ui_inbox_lock:
                actions = list(pending_ui_actions)
                pending_ui_actions.clear()
                messages = list(pending_messages)
                pending_messages.clear()
                frame_payload = pending_frame["payload"]
                pending_frame["payload"] = None

            # Acciones de precarga, mensajes y frame se aplican siempre en el
            # mismo event loop que administra los controles de la página
            for action in actions:
                try:
                    action()
                except Exception:
                    pass
            for message in messages:
                _apply_info(message)
            if frame_payload is not None:
                _apply_frame_with_counts(*frame_payload)
            elif (
                detection_state["is_playing"]
                and frame_state["first_frame_received"]
                and frame_state["fps_last_frame_at"] > 0.0
                and time.perf_counter() - frame_state["fps_last_frame_at"] >= 1.5
                and not frame_state["fps_stalled"]
            ):

                frame_state["fps_stalled"] = True
                set_fps_indicator(0.0)
                fps_indicator.update()

            if refresh_processing_timer():

                processing_timer.update()

            await asyncio.sleep(1.0 / 30.0)

    def create_processor():

        return VideoProcessor(
            on_frame=on_frame_with_counts,
            on_info=on_info,
            enable_vehicles=detection_state["enable_vehicles"],
            enable_lanes=detection_state["enable_lanes"],
        )

    # Crear procesador inicial
    processor_ref["processor"] = create_processor()

    selected_file_path = {"path": None, "name": None}
    action_buttons = []

    def set_action_buttons_enabled(enabled: bool, allow_upload: bool = False):
        # Bloquea o desbloquea Subir/Iniciar/Parar/Limpiar. Se usa especialmente
        # durante precarga para que el usuario no arranque sin modelos listos.
        for index, action_button in enumerate(action_buttons):
            # Elegir el archivo puede hacerse mientras la GPU se prepara; así el
            # tiempo de interacción del usuario se solapa con la precarga.
            action_button.disabled = not enabled and not (allow_upload and index == 0)
        update_switches_state()
        try:
            page.update()
        except Exception:
            pass

    def start_preload():
        # Carga YOLO en segundo plano sin ocultar la pantalla principal 
        if app_state["closing"]:
            return
        detection_state["models_ready"] = False
        detection_state["is_preloading"] = True
        status_indicator.content.controls[0].color = COLORS.ORANGE
        status_indicator.content.controls[1].value = "Preparando GPU"
        set_action_buttons_enabled(False, allow_upload=True)

        def apply_preload_done(ok: bool):

            if app_state["closing"]:
                return
            detection_state["models_ready"] = ok
            detection_state["is_preloading"] = False
            if (
                not frame_state.get("first_frame_received")
                and not selected_file_path["path"]
            ):
                video_panel.content = video_placeholder

            set_action_buttons_enabled(True)
            status_indicator.content.controls[0].color = (
                COLORS.RED if not ok else COLORS.GREEN
            )
            if detection_state["is_playing"] and ok:
                status_indicator.content.controls[1].value = "Procesando"
            else:
                status_indicator.content.controls[1].value = (
                    "Error de precarga" if not ok else "Listo"
                )

        def on_preload_done(ok: bool):
            # `_ensure_detector_loaded` llama este callback desde su worker. La
            # mutación real se difiere al pump para no bloquear o congelar Flet
            enqueue_ui_action(lambda result=ok: apply_preload_done(result))

        if processor_ref["processor"]:
            processor_ref["processor"].preload_models(on_done=on_preload_done)

    def stop_current_processor() -> bool:

        if not processor_ref["processor"]:
            return True

        if processor_ref["processor"].stop():
            return True

        if app_state["closing"]:
            return False

        alert_row = ft.Row(
            [
                ft.Icon(ICONS.WARNING, size=16, color=COLORS.ORANGE),
                ft.Text(
                    "Espere: cerrando inferencia actual", color=COLORS.BLACK_87, size=12
                ),
            ],
            spacing=6,
        )
        alerts.controls.insert(0, alert_row)
        page.update()
        return False

    cleanup_state = {"running": False, "done": False}

    def cleanup_processor() -> bool:

        if cleanup_state["running"] or cleanup_state["done"]:
            return True

        app_state["closing"] = True
        cleanup_state["running"] = True
        released = True
        processor = processor_ref.get("processor")
        try:
            if processor:
                processor.on_frame = lambda *_args, **_kwargs: None
                processor.on_info = lambda *_args, **_kwargs: None
                released = processor.release_models(timeout=45.0)
            if released:
                cleanup_state["done"] = True
            return released
        finally:
            cleanup_state["running"] = False

    atexit.register(cleanup_processor)

    async def on_window_event(e):

        if getattr(e, "type", None) == ft.WindowEventType.CLOSE:
            cleanup_processor()
            try:
                await page.window.destroy()
            except Exception:
                pass

    async def on_page_close(e):
        cleanup_processor()

    try:
        page.window.prevent_close = True
        page.window.on_event = on_window_event
    except Exception:
        pass

    try:
        page.on_close = on_page_close
        page.on_disconnect = on_page_close
    except Exception:
        pass

    def handle_picked_files(files):
        # Procesa el resultado asíncrono del selector moderno de Flet 
        if not files:
            return
        f = files[0]

        # Detener procesador anterior si existe
        if not stop_current_processor():
            return
        discard_pending_frame()

        selected_file_path["path"] = getattr(f, "path", None)
        selected_file_path["name"] = f.name

        # Resetear estado del frame y guardar nombre del archivo
        frame_state["first_frame_received"] = False
        frame_state["current_filename"] = f.name
        reset_fps_metric()
        reset_processing_timer()

        alert_row = ft.Row(
            [
                ft.Icon(ICONS.CHECK_CIRCLE, size=16, color=COLORS.GREEN),
                ft.Text(f"Video cargado: {f.name}", color=COLORS.BLACK_87, size=12),
            ],
            spacing=6,
        )
        alerts.controls.insert(0, alert_row)

        # Mostrar placeholder de cargando mientras se inicializan los detectores
        video_panel.content = ft.Column(
            [
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Icon(ICONS.VIDEO_FILE, size=16, color=COLORS.WHITE_54),
                            ft.Text(
                                f.name,
                                color=COLORS.WHITE,
                                size=12,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                        ],
                        spacing=6,
                    ),
                    padding=ft.Padding.only(bottom=8),
                ),
                ft.Container(
                    content=loading_placeholder,
                    expand=True,
                    alignment=ft.Alignment.CENTER,
                ),
            ],
            expand=True,
        )

        # Marcar como reproduciendo y deshabilitar switches
        detection_state["is_playing"] = True
        update_switches_state()

        # Resetear velocidad a 1.0x
        speed_text.value = "1.0x"

        status_indicator.content.controls[0].color = COLORS.GREEN
        status_indicator.content.controls[1].value = "Procesando"

        page.update()

        if selected_file_path["path"]:
            started = processor_ref["processor"].start(selected_file_path["path"])
            if not started:
                detection_state["is_playing"] = False
                update_switches_state()
                status_indicator.content.controls[0].color = COLORS.RED
                status_indicator.content.controls[1].value = "Detenido"
                page.update()
            else:
                start_processing_timer()
                processing_timer.update()

    # === ACCIONES DE BOTONES ===
    async def do_upload(_):
        files = await ft.FilePicker().pick_files(
            allow_multiple=False,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["mp4", "mov", "avi", "mkv", "webm"],
        )
        handle_picked_files(files)

    def do_start(_):
        # Si ya está reproduciendo, no reiniciar el archivo ni recrear workers
        # El botón se vuelve ante clics repetidos.
        if detection_state["is_playing"]:
            return

        if not selected_file_path["path"]:
            alert_row = ft.Row(
                [
                    ft.Icon(ICONS.WARNING, size=16, color=COLORS.ORANGE),
                    ft.Text(
                        "Cargue un video antes de iniciar",
                        color=COLORS.BLACK_87,
                        size=12,
                    ),
                ],
                spacing=6,
            )
            alerts.controls.insert(0, alert_row)
            page.update()
        else:
            reset_fps_metric()
            started = processor_ref["processor"].start(selected_file_path["path"])
            if not started:
                return
            start_processing_timer()

            # Marcar como reproduciendo y deshabilitar switches
            detection_state["is_playing"] = True
            update_switches_state()

            status_indicator.content.controls[0].color = COLORS.GREEN
            status_indicator.content.controls[1].value = "Procesando"
            page.update()

    def do_stop(_):
        if not stop_current_processor():
            return
        discard_pending_frame()
        reset_fps_metric()
        stop_processing_timer()

        # Marcar como detenido y habilitar switches
        detection_state["is_playing"] = False
        update_switches_state()

        status_indicator.content.controls[0].color = COLORS.RED
        status_indicator.content.controls[1].value = "Detenido"
        count_pesados.value = "0"
        count_livianos.value = "0"
        count_motos.value = "0"

        alert_row = ft.Row(
            [
                ft.Icon(ICONS.STOP, size=16, color=COLORS.BLUE),
                ft.Text("Análisis detenido", color=COLORS.BLACK_87, size=12),
            ],
            spacing=6,
        )
        alerts.controls.insert(0, alert_row)
        page.update()

    def do_clear(_):
        #Limpia la sesión sin descargar otra vez los modelos de la GPU
        discard_pending_frame()
        detection_state["is_playing"] = False
        detection_state["is_preloading"] = False
        frame_state["first_frame_received"] = False
        frame_state["current_filename"] = ""
        frame_state["last_stats_update"] = 0.0
        reset_fps_metric()
        reset_processing_timer()

        # Detener el pipeline y resetear caches/estadísticas
        processor = processor_ref["processor"]
        if processor and not processor.reset_session():
            return
        detection_state["models_ready"] = bool(
            processor and processor.models_ready()
        )

        #Marcar como detenido y habilitar switches
        detection_state["is_playing"] = False
        update_switches_state()

        # Limpiar alertas
        alerts.controls.clear()

        #resetear contadores
        count_pesados.value = "0"
        count_livianos.value = "0"
        count_motos.value = "0"

        #Limpiar imagen
        image_view.src = ""

        # Restaurar el panel inicial
        video_panel.content = video_placeholder

        #  Resetear estado
        status_indicator.content.controls[0].color = COLORS.RED
        status_indicator.content.controls[1].value = "Detenido"

        #Resetear velocidad
        speed_text.value = "1.0x"

        #Limpiar path seleccionado
        selected_file_path["path"] = None
        selected_file_path["name"] = None

        # Conservar los modelos cargados
        if detection_state["models_ready"]:
            set_action_buttons_enabled(True)
        else:
            start_preload()

        #Agregar un único mensaje de limpieza
        alert_row = ft.Row(
            [
                ft.Icon(ICONS.CLEANING_SERVICES, size=16, color=COLORS.GREY),
                ft.Text(
                    "Sesión limpia; modelos conservados en GPU",
                    color=COLORS.BLACK_87,
                    size=12,
                ),
            ],
            spacing=6,
        )
        alerts.controls.insert(0, alert_row)

        page.update()

    def do_help(_):
        def close_dialog(_):
            page.pop_dialog()
            page.update()

        dialog = ft.AlertDialog(
            title=ft.Text("📖 Ayuda - Detección de Vehículos"),
            content=ft.Column(
                [
                    ft.Text("Cómo usar:", weight=ft.FontWeight.BOLD),
                    ft.Text("1. 'Subir' → cargar video"),
                    ft.Text("2. El análisis inicia automáticamente"),
                    ft.Text("3. 'Parar' → detener"),
                    ft.Text("4. 'Iniciar' → reanudar"),
                    ft.Text("5. 'Limpiar' → reiniciar sin recargar modelos"),
                    ft.Divider(),
                    ft.Text("Opciones:", weight=ft.FontWeight.BOLD),
                    ft.Text("• Switch Vehículos: activa/desactiva YOLO"),
                    ft.Text("• Switch Carriles: activa/desactiva líneas"),
                    ft.Text("• ⚠ Detener video para cambiar opciones"),
                    ft.Divider(),
                    ft.Text(
                        "Modelos: YOLO11 OBB nano + YOLO11 Seg small",
                        size=12,
                        color=COLORS.GREY,
                    ),
                ],
                spacing=8,
                tight=True,
            ),
            actions=[ft.TextButton("Cerrar", on_click=close_dialog)],
        )
        page.show_dialog(dialog)
        page.update()

    # === LAYOUT ===
    upload_button = btn("Subir", BTN_BLUE, do_upload, icon=ICONS.UPLOAD_FILE)
    start_button = btn("Iniciar", BTN_GREEN, do_start, icon=ICONS.PLAY_ARROW)
    stop_button = btn("Parar", BTN_RED, do_stop, icon=ICONS.STOP)
    clear_button = btn("Limpiar", BTN_GRAY, do_clear, icon=ICONS.CLEANING_SERVICES)
    help_button = btn("?", BTN_ORANGE, do_help)
    action_buttons.extend([upload_button, start_button, stop_button, clear_button])

    topbar = ft.Container(
        bgcolor=TOPBAR_BG,
        padding=ft.Padding.symmetric(horizontal=16, vertical=10),
        content=ft.Row(
            [
                ft.Row(
                    [
                        ft.Icon(ICONS.DIRECTIONS_CAR, color=COLORS.WHITE, size=24),
                        ft.Text(
                            "Detección de Vehículos",
                            color=COLORS.WHITE,
                            weight=ft.FontWeight.BOLD,
                            size=20,
                        ),
                    ],
                    spacing=10,
                ),
                ft.Row(
                    [
                        status_indicator,
                        fps_indicator,
                        speed_control,
                        processing_timer,
                    ],
                    spacing=12,
                ),
                ft.Row(
                    [
                        upload_button,
                        start_button,
                        stop_button,
                        clear_button,
                        help_button,
                    ],
                    spacing=12,
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )

    sidebar = ft.Container(
        width=260,
        bgcolor=SIDEBAR_BG,
        border_radius=16,
        padding=10,
        content=ft.Column(
            [
                ft.Text(
                    "Detecciones Actuales",
                    size=14,
                    weight=ft.FontWeight.BOLD,
                    color=COLORS.BLACK_87,
                ),
                ft.Container(height=4),
                counters_row,
                ft.Divider(height=12),
                ft.Text(
                    "Estado del Carril",
                    size=14,
                    weight=ft.FontWeight.BOLD,
                    color=COLORS.BLACK_87,
                ),
                ft.Container(height=4),
                lane_status_row,
                lane_deviation_text,
                ft.Divider(height=12),
                detection_controls,
                ft.Divider(height=12),
                ft.Row(
                    [
                        ft.Icon(ICONS.NOTIFICATIONS, size=18, color=COLORS.BLACK_87),
                        ft.Text(
                            "Alertas",
                            size=16,
                            weight=ft.FontWeight.BOLD,
                            color=COLORS.BLACK_87,
                        ),
                    ],
                    spacing=6,
                ),
                ft.Container(height=4),
                alerts,
            ],
            expand=True,
            spacing=4,
        ),
    )

    body = ft.Container(
        expand=True,
        padding=12,
        content=ft.Row(
            controls=[sidebar, ft.Container(width=12), video_panel], expand=True
        ),
    )

    root = ft.Column([topbar, body], spacing=0, expand=True)
    page.add(root)
    page.run_task(ui_event_pump)
    start_preload()


if __name__ == "__main__":
    ft.run(main)
