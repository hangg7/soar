import dataclasses
import sys
from threading import Lock
from typing import Any, Callable, Literal, Optional

import numpy as np
from jaxtyping import Float32


@dataclasses.dataclass
class CameraState(object):
    fov: float
    aspect: float
    c2w: Float32[np.ndarray, "4 4"]
    extras: dict[str, Any]
    mode: Literal["RGB", "Depth", "Normals", "Depth Normals"] = "RGB"


@dataclasses.dataclass
class LightState(object):
    direction: Float32[np.ndarray, "3"]
    color: Float32[np.ndarray, "3"]


@dataclasses.dataclass
class ViewerStats(object):
    num_train_rays_per_sec: Optional[float] = None
    num_view_rays_per_sec: float = 100000.0


class InterruptRenderException(Exception):
    pass


class set_trace_context(object):
    def __init__(self, func):
        self.func = func

    def __enter__(self):
        sys.settrace(self.func)
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        sys.settrace(None)


view_lock = Lock()


def with_view_lock(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        with view_lock:
            return fn(*args, **kwargs)

    return wrapper
