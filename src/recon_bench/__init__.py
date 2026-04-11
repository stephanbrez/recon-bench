"""
recon-bench: 3D reconstruction benchmarking toolkit.

Evaluate reconstruction quality with a single call:

    import recon_bench

    # Image vs image
    result = recon_bench.evaluate(Path("gt.png"), Path("pred.png"))
    print(result.image_metrics)   # {"psnr": 32.4, "ssim": 0.91, ...}

    # Mesh vs mesh (chamfer distance)
    result = evaluate(Path("gt.obj"), Path("pred.obj"))
    print(result.geometry_metrics)  # {"chamfer_distance": 0.002}

    # Mesh vs image (render mesh, compare to reference)
    result = evaluate(Path("ref.png"), Path("model.obj"), camera=Camera.orbit())

    # Mesh vs mesh with image evaluation
    result = evaluate(Path("gt.obj"), Path("pred.obj"), image_eval=True)
"""
from . import evaluate as _evaluate_module
from . import _types
from .profiling import _types as _profile_mod
from .profiling import timer as _timer_mod
from .profiling import memory as _memory_mod

evaluate = _evaluate_module.evaluate
Camera = _types.Camera
EvalResult = _types.EvalResult
GeometryArrays = _types.GeometryArrays
GeometryType = _types.GeometryType
ProfileResult = _profile_mod.ProfileResult
TimingEntry = _profile_mod.TimingEntry
MemoryEntry = _profile_mod.MemoryEntry
Timer = _timer_mod.Timer
MemoryTracker = _memory_mod.MemoryTracker
