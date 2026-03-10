from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from aether_ml import AircraftDetectionPipeline, ChangeDetectionOnnxPipeline, ViTAircraftClassifierPipeline

from ml_inference.geo_projection import GeoContext, geo_context_from_bounds, read_geotiff_with_context
from ml_inference.output import write_events, write_result
from ml_inference.pipeline import PipelineModels, run_intelligence


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _load_image_with_geo(path: Path) -> tuple[np.ndarray, GeoContext | None]:
    if path.suffix.lower() in {".tif", ".tiff"}:
        img, geo_ctx = read_geotiff_with_context(path)
        return img, geo_ctx
    return _read_image(path), None


def _build_classifier(weights_path: str, num_classes: int, model_name: str, image_size: int) -> ViTAircraftClassifierPipeline:
    return ViTAircraftClassifierPipeline(
        weights_path=weights_path,
        num_classes=num_classes,
        model_name=model_name,
        image_size=image_size,
        device=None,
    )


def _classifier_fn(pipeline: ViTAircraftClassifierPipeline):
    def _fn(img: np.ndarray) -> dict[str, Any]:
        res = pipeline.classify(img)
        return {
            "class_id": res.class_id,
            "class_name": res.class_name,
            "confidence": res.confidence,
            "origin_country": res.origin_country,
        }
    return _fn


def _load_sidecar(sidecar_path: Path) -> GeoContext | None:
    if not sidecar_path.is_file():
        return None
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    transform = payload.get("transform")
    crs = payload.get("crs")
    width = int(payload.get("width", 0))
    height = int(payload.get("height", 0))
    bounds = payload.get("tile_bounds")
    tile_id = payload.get("tile_id")
    if bounds is not None:
        bounds = (bounds[0], bounds[1], bounds[2], bounds[3])
    if not transform or width <= 0 or height <= 0:
        return None
    return GeoContext(transform=transform, crs=crs, width=width, height=height, bounds=bounds, tile_id=tile_id)


def _pair_batch_files(input_dir: Path) -> list[tuple[Path, Path, Path | None]]:
    pairs: list[tuple[Path, Path, Path | None]] = []
    t1_files = sorted(input_dir.glob("*_t1.*"))
    for t1 in t1_files:
        base = t1.stem.replace("_t1", "")
        t2 = input_dir / t1.name.replace("_t1", "_t2")
        if not t2.exists():
            continue
        sidecar = input_dir / f"{base}.json"
        pairs.append((t1, t2, sidecar if sidecar.exists() else None))
    return pairs


def _load_models(args: argparse.Namespace) -> PipelineModels:
    change_detector = ChangeDetectionOnnxPipeline(args.change_onnx, device="auto")
    aircraft_detector = AircraftDetectionPipeline(
        args.aircraft_onnx,
        confidence_threshold=args.aircraft_conf_threshold,
        iou_threshold=args.aircraft_iou_threshold,
        device="auto",
    )
    classifier = None
    if args.classifier_weights:
        classifier = _build_classifier(
            args.classifier_weights,
            args.classifier_num_classes,
            args.classifier_model_name,
            args.classifier_image_size,
        )
    return PipelineModels(
        change_detector=change_detector,
        aircraft_detector=aircraft_detector,
        classifier_fn=_classifier_fn(classifier) if classifier else None,
    )


def run_single(args: argparse.Namespace) -> dict[str, Any]:
    before_path = Path(args.before)
    after_path = Path(args.after)

    before_img, geo_ctx_before = _load_image_with_geo(before_path)
    after_img, geo_ctx_after = _load_image_with_geo(after_path)

    geo_ctx = geo_ctx_after or geo_ctx_before
    if geo_ctx is None and args.geo_bounds:
        geo_ctx = geo_context_from_bounds(after_img.shape[1], after_img.shape[0], args.geo_bounds)

    models = _load_models(args)
    return run_intelligence(
        before_img,
        after_img,
        geo_ctx,
        models=models,
        run_change_detection=args.run_change_detection,
        run_aircraft_detection=args.run_aircraft_detection,
        max_detections=args.max_detections,
        change_threshold=args.change_threshold,
    )


def run_batch(args: argparse.Namespace) -> list[dict[str, Any]]:
    input_dir = Path(args.input_dir)
    pairs = _pair_batch_files(input_dir)
    models = _load_models(args)
    results: list[dict[str, Any]] = []
    for before_path, after_path, sidecar in pairs:
        geo_ctx = _load_sidecar(sidecar) if sidecar else None
        if before_path.suffix.lower() in {".tif", ".tiff"}:
            before_img, before_geo = read_geotiff_with_context(before_path)
            if geo_ctx is None:
                geo_ctx = before_geo
        else:
            before_img = _read_image(before_path)
        if after_path.suffix.lower() in {".tif", ".tiff"}:
            after_img, after_geo = read_geotiff_with_context(after_path)
            if geo_ctx is None:
                geo_ctx = after_geo
        else:
            after_img = _read_image(after_path)
        if geo_ctx is None and args.geo_bounds:
            geo_ctx = geo_context_from_bounds(after_img.shape[1], after_img.shape[0], args.geo_bounds)

        out = run_intelligence(
            before_img,
            after_img,
            geo_ctx,
            models=models,
            run_change_detection=args.run_change_detection,
            run_aircraft_detection=args.run_aircraft_detection,
            max_detections=args.max_detections,
            change_threshold=args.change_threshold,
        )
        results.append(out)
    return results


def run_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = json.loads(Path(args.tile_metadata).read_text(encoding="utf-8"))
    models = _load_models(args)
    results: list[dict[str, Any]] = []
    for job in manifest.get("jobs", []):
        before_path = Path(job["before"])
        after_path = Path(job["after"])
        sidecar_path = Path(job["sidecar"]) if "sidecar" in job else None

        geo_ctx = _load_sidecar(sidecar_path) if sidecar_path else None
        if before_path.suffix.lower() in {".tif", ".tiff"}:
            before_img, before_geo = read_geotiff_with_context(before_path)
            if geo_ctx is None:
                geo_ctx = before_geo
        else:
            before_img = _read_image(before_path)
        if after_path.suffix.lower() in {".tif", ".tiff"}:
            after_img, after_geo = read_geotiff_with_context(after_path)
            if geo_ctx is None:
                geo_ctx = after_geo
        else:
            after_img = _read_image(after_path)
        if geo_ctx is None and args.geo_bounds:
            geo_ctx = geo_context_from_bounds(after_img.shape[1], after_img.shape[0], args.geo_bounds)

        out = run_intelligence(
            before_img,
            after_img,
            geo_ctx,
            models=models,
            run_change_detection=args.run_change_detection,
            run_aircraft_detection=args.run_aircraft_detection,
            max_detections=args.max_detections,
            change_threshold=args.change_threshold,
        )
        results.append(out)
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aether-Eye inference runner")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--before", type=str, help="Before image path (single-pair mode).")
    mode.add_argument("--input_dir", type=str, help="Batch directory containing *_t1/_t2 pairs.")
    mode.add_argument("--tile_metadata", type=str, help="Manifest JSON with jobs list.")

    p.add_argument("--after", type=str, help="After image path (single-pair mode).")
    p.add_argument("--geo_bounds", type=float, nargs=4, metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"))
    p.add_argument("--no_change_detection", action="store_true", default=False)
    p.add_argument("--no_aircraft_detection", action="store_true", default=False)
    p.add_argument("--max_detections", type=int, default=25)
    p.add_argument("--change_threshold", type=float, default=0.5)
    p.add_argument("--output", type=str, default="output/events.json")
    p.add_argument("--format", type=str, default="json", choices=["json", "geojson"])

    p.add_argument("--change_onnx", type=str, required=True, help="Path to change detection ONNX model.")
    p.add_argument("--aircraft_onnx", type=str, required=True, help="Path to aircraft detection ONNX model.")
    p.add_argument("--aircraft_conf_threshold", type=float, default=0.25)
    p.add_argument("--aircraft_iou_threshold", type=float, default=0.45)

    p.add_argument("--classifier_weights", type=str, default="")
    p.add_argument("--classifier_num_classes", type=int, default=100)
    p.add_argument("--classifier_model_name", type=str, default="vit_base_patch16_224")
    p.add_argument("--classifier_image_size", type=int, default=224)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.run_change_detection = not args.no_change_detection
    args.run_aircraft_detection = not args.no_aircraft_detection
    if args.before:
        if not args.after:
            raise SystemExit("--after is required with --before.")
        result = run_single(args)
        write_result(result, args.output, fmt=args.format)
    elif args.input_dir:
        results = run_batch(args)
        # Flatten for output
        events = [ev for r in results for ev in r["events"]]
        if args.format == "json":
            summary = {"batches": len(results), "events": len(events)}
            processing = {"event_count": len(events)}
            write_result({"events": events, "summary": summary, "processing": processing}, args.output, fmt=args.format)
        else:
            write_events(events, args.output, fmt=args.format)
    else:
        results = run_manifest(args)
        events = [ev for r in results for ev in r["events"]]
        if args.format == "json":
            summary = {"jobs": len(results), "events": len(events)}
            processing = {"event_count": len(events)}
            write_result({"events": events, "summary": summary, "processing": processing}, args.output, fmt=args.format)
        else:
            write_events(events, args.output, fmt=args.format)


if __name__ == "__main__":
    main()
