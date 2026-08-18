
#No admite inferencia en CPU 


from __future__ import annotations

from functools import lru_cache

import torch


GPU_NAME_TOKEN = "RTX 5050"
MIN_COMPUTE_CAPABILITY = (12, 0)


class RequiredGpuError(RuntimeError):
    pass


def _available_cuda_devices() -> list[tuple[int, str]]:
    #Devuelve los dispositivos visibles para CUDA con su índice y nombre
    return [
        (index, torch.cuda.get_device_name(index))
        for index in range(torch.cuda.device_count())
    ]


def _configure_torch_for_inference() -> None:
    #Activa rutas rápidas seguras para inferencia con formas de entrada fijas
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # PyTorch moderno prefiere fp32_precision sobre allow_tf32
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except (AttributeError, RuntimeError):
        torch.set_float32_matmul_precision("high")


@lru_cache(maxsize=1)
def require_rtx_5050() -> dict:
    #Selecciona la RTX 5050 y devuelve información inmutable de ejecución
    if not torch.backends.cuda.is_built():
        raise RequiredGpuError(
            "PyTorch fue instalado sin CUDA. Reinstale requirements.txt desde "
            "el índice oficial cu130."
        )

    if not torch.cuda.is_available():
        raise RequiredGpuError(
            "CUDA no está disponible. Actualice el controlador NVIDIA y confirme "
            "que instaló PyTorch 2.13 con CUDA 13.0."
        )

    devices = _available_cuda_devices()
    selected = next(
        (
            (index, name)
            for index, name in devices
            if GPU_NAME_TOKEN.casefold() in name.casefold()
        ),
        None,
    )
    if selected is None:
        visible = ", ".join(f"cuda:{i}={name}" for i, name in devices) or "ninguno"
        raise RequiredGpuError(
            f"No se encontró una NVIDIA GeForce {GPU_NAME_TOKEN}. "
            f"Dispositivos CUDA visibles: {visible}."
        )

    index, device_name = selected
    torch.cuda.set_device(index)
    props = torch.cuda.get_device_properties(index)
    capability = (props.major, props.minor)
    if capability < MIN_COMPUTE_CAPABILITY:
        raise RequiredGpuError(
            f"{device_name} reportó compute capability {props.major}.{props.minor}; "
            "la RTX 5050 debe reportar 12.0 o superior."
        )

    if not torch.version.cuda:
        raise RequiredGpuError("La distribución instalada de PyTorch no incluye CUDA.")

    _configure_torch_for_inference()
    return {
        "cuda_available": True,
        "device": f"cuda:{index}",
        "device_index": index,
        "device_name": device_name,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "compute_capability": f"{props.major}.{props.minor}",
        "memory_total_gb": props.total_memory / (1024**3),
    }


def current_gpu_memory() -> dict:
    #Consulta la VRAM de la GPU seleccionada sin cambiar de dispositivo
    info = require_rtx_5050()
    index = info["device_index"]
    return {
        "allocated_gb": torch.cuda.memory_allocated(index) / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved(index) / (1024**3),
        "total_gb": info["memory_total_gb"],
    }
