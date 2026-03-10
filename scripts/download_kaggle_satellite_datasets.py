import argparse
import logging
import subprocess
import sys
import zipfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download SpaceNet-7 and DOTA datasets via Kaggle API (Hub or CLI) into Aether-Eye datasets."
    )
    p.add_argument(
        "--target-root",
        type=Path,
        default=Path("data/raw"),
        help="Root folder where datasets will be copied (relative to project root).",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Force download even if destination already exists.",
    )
    p.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Additional Kaggle dataset to download. Format: name=kaggle_id or kaggle_id.",
    )
    return p.parse_args()


def _parse_dataset_args(entries: list[str]) -> dict[str, str]:
    datasets: dict[str, str] = {}
    for raw in entries:
        if not raw:
            continue
        if "=" in raw:
            name, kaggle_id = raw.split("=", 1)
            name = name.strip()
            kaggle_id = kaggle_id.strip()
            if name and kaggle_id:
                datasets[name] = kaggle_id
        else:
            kaggle_id = raw.strip()
            if kaggle_id:
                datasets[kaggle_id.split("/")[-1]] = kaggle_id
    return datasets


def download_with_hub(kaggle_id: str, dest_path: Path, force: bool = False) -> bool:
    """Attempts to download using the kagglehub library."""
    try:
        import kagglehub
    except ImportError:
        return False

    if dest_path.exists() and any(dest_path.iterdir()) and not force:
        logger.info(f"Dataset already exists at {dest_path}. Skipping.")
        return True

    logger.info(f"Downloading {kaggle_id} via kagglehub...")
    try:
        src_path = Path(kagglehub.dataset_download(kaggle_id)).resolve()
        dest_path.mkdir(parents=True, exist_ok=True)

        if src_path.is_file() and src_path.suffix == ".zip":
            with zipfile.ZipFile(src_path, "r") as zip_ref:
                zip_ref.extractall(dest_path)
            logger.info(f"Extracted {src_path} to {dest_path}")
        elif src_path.is_dir():
            # Create a pointer or copy files
            (dest_path / "SOURCE_PATH.txt").write_text(str(src_path), encoding="utf-8")
            logger.info(f"Dataset located at {src_path}. Pointer created at {dest_path}")
            
            # Simple copy of top-level files to make it "look" like it's there
            # Real intelligence systems might symlink, but pointers are safer for initial setup
            
        return True
    except Exception as e:
        logger.error(f"KaggleHub download failed: {e}")
        return False


def download_with_cli(kaggle_id: str, dest_path: Path, force: bool = False) -> bool:
    """Attempts to download using the kaggle CLI."""
    if dest_path.exists() and any(dest_path.iterdir()) and not force:
        return True

    logger.info(f"Attempting download of {kaggle_id} via Kaggle CLI...")
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Try 'kaggle' directly, then 'python -m kaggle'
    commands = [
        ["kaggle", "datasets", "download", "-d", kaggle_id, "-p", str(dest_path), "--unzip"],
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", kaggle_id, "-p", str(dest_path), "--unzip"]
    ]

    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                logger.info(f"Successfully downloaded {kaggle_id} using {' '.join(cmd[:2])}")
                return True
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.debug(f"Command {' '.join(cmd)} failed: {e}")
            continue

    return False


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    target_root = (repo_root / args.target_root).resolve()

    datasets = {
        "spacenet": "amerii/spacenet-7-multitemporal-urban-development",
        "dota": "chandlertimm/dota-data",
    }
    datasets.update(_parse_dataset_args(args.dataset))

    for name, kaggle_id in datasets.items():
        dest = target_root / name
        
        # Try Hub first, then CLI
        success = download_with_hub(kaggle_id, dest, force=args.force_download)
        if not success:
            success = download_with_cli(kaggle_id, dest, force=args.force_download)
            
        if success:
            logger.info(f"Successfully prepared {name} dataset.")
        else:
            logger.error(f"Failed to prepare {name} dataset.")
            logger.info("Check: 1. Internet 2. Kaggle API credentials 3. Dependencies (pip install kagglehub kaggle)")


if __name__ == "__main__":
    main()
