from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import importlib.util
import os


REPO_ROOT = Path(__file__).resolve().parents[1]
STANFORD_ROOT = REPO_ROOT / "VisionBasedAircraftDAA"
STANFORD_DATA_GEN = STANFORD_ROOT / "src" / "data_generation"
STANFORD_DATASETS = STANFORD_ROOT / "datasets"


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)} (cwd={cwd})")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _xplane_preflight(timeout_seconds: float = 2.0) -> None:
    """
    Fail fast with actionable guidance when X-Plane/XPlaneConnect is not reachable.
    """
    xpc_path = STANFORD_ROOT / "src" / "xpc3.py"
    if not xpc_path.exists():
        raise FileNotFoundError(f"Missing XPlaneConnect client file: {xpc_path}")

    spec = importlib.util.spec_from_file_location("xpc3_local", str(xpc_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {xpc_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    client = None
    try:
        # Defaults match the Stanford repo's xpc3.py constructor.
        client = mod.XPlaneConnect(timeout=int(timeout_seconds * 1000))
        client.getDREF("sim/version/xplane_internal_version")[0]
    except Exception as exc:
        xp_host = os.environ.get("XPC_XP_HOST", "localhost")
        xp_port = os.environ.get("XPC_XP_PORT", "49009")
        raise RuntimeError(
            "X-Plane preflight failed. Stanford generator requires a live X-Plane instance "
            f"with XPlaneConnect plugin reachable on UDP (host={xp_host}, port={xp_port}).\n"
            "Checklist:\n"
            "  1) Start X-Plane before running this script.\n"
            "  2) Ensure XPlaneConnect plugin is installed/enabled.\n"
            "  3) Allow X-Plane/UDP through Windows Firewall.\n"
            "  4) If your plugin listens on a different port, update "
            "VisionBasedAircraftDAA/src/xpc3.py defaults (xpPort).\n"
            f"Original error: {exc}"
        ) from exc
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wrapper around VisionBasedAircraftDAA data generator."
    )
    p.add_argument("--dataset-name", default="stanford_military_500")
    p.add_argument("--craft", default="King Air C90")
    p.add_argument("--train", type=int, default=450)
    p.add_argument("--valid", type=int, default=50)
    p.add_argument("--location", default="Palo Alto")
    p.add_argument("--weather", type=int, default=0)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "stanford_aircraft",
        help="Target folder in Aether-Eye where generated dataset is copied.",
    )
    p.add_argument(
        "--skip-copy",
        action="store_true",
        help="Only run Stanford generator and leave outputs in VisionBasedAircraftDAA/datasets.",
    )
    p.add_argument(
        "--newac",
        action="store_true",
        help="Pass --newac to Stanford generator (will prompt in terminal).",
    )
    p.add_argument(
        "--xp-host",
        default="localhost",
        help="X-Plane host for XPlaneConnect (default: localhost).",
    )
    p.add_argument(
        "--xp-port",
        type=int,
        default=49009,
        help="X-Plane UDP port for XPlaneConnect (try 49007 if 49009 fails).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not STANFORD_DATA_GEN.exists():
        raise FileNotFoundError(f"Missing Stanford generator: {STANFORD_DATA_GEN}")

    os.environ["XPC_XP_HOST"] = args.xp_host
    os.environ["XPC_XP_PORT"] = str(args.xp_port)

    _xplane_preflight()

    cmd = [
        sys.executable,
        "-m",
        "generate_traffic_data",
        "--name",
        args.dataset_name,
        "-ac",
        args.craft,
        "--train",
        str(args.train),
        "--valid",
        str(args.valid),
        "--location",
        args.location,
        "--weather",
        str(args.weather),
    ]
    if args.newac:
        cmd.append("--newac")

    try:
        _run(cmd, cwd=STANFORD_DATA_GEN)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Stanford generator failed. If traceback shows WinError 10054/timeout, "
            "the UDP connection to X-Plane dropped. Re-check X-Plane runtime and firewall."
        ) from exc

    source = STANFORD_DATASETS / args.dataset_name
    if not source.exists():
        raise FileNotFoundError(f"Stanford dataset output missing: {source}")

    if args.skip_copy:
        print(f"[ok] generated at: {source}")
        return

    target = args.output_dir / args.dataset_name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    print(f"[ok] copied Stanford output to: {target}")


if __name__ == "__main__":
    main()
