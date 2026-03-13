"""Route package.

Keep package imports side-effect free so optional route modules can be loaded
independently by `app.main`.
"""

__all__ = [
    "health",
    "inference",
    "onnx_inference",
    "vit_explainability",
    "aircraft_inference",
    "change_inference",
    "intelligence",
    "live_aircraft",
    "operations",
]
