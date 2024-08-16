from packaging.version import Version

import threestudio

if hasattr(threestudio, "__version__") and Version(threestudio.__version__) >= Version(
    "0.2.0"
):
    pass
else:
    if hasattr(threestudio, "__version__"):
        print(f"[INFO] threestudio version: {threestudio.__version__}")
    raise ValueError(
        "threestudio version must be >= 0.2.0, please update threestudio by pulling the latest version from github"
    )


from .background import gaussian_mvdream_background
from .data import uncond_multiview
from .geometry import exporter, gaussian_base, gaussian_io, surfel_base
from .guidance import imagedream_guidance, mvdream_guidance
from .renderer import diff_gaussian_rasterizer
from .system import gaussian_mvdream, gaussian_splatting, gaussian_surfel_mvdream
from .utils import smpl
