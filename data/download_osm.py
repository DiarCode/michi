"""Download and cache Kazakhstan OSM extract for Astana parsing."""

import os
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

OSM_URL = "https://download.geofabrik.de/asia/kazakhstan-latest.osm.pbf"
CACHE_FILE = CACHE_DIR / "kazakhstan-latest.osm.pbf"

ASTANA_BBOX = {
    "west": 71.25,
    "east": 71.65,
    "south": 50.95,
    "north": 51.25,
}


def download_osm(force: bool = False) -> Path:
    """Download Kazakhstan OSM PBF if not cached or stale."""
    if CACHE_FILE.exists() and not force:
        import time
        age_days = (time.time() - CACHE_FILE.stat().st_mtime) / 86400
        if age_days < 30:
            print(f"Using cached OSM file: {CACHE_FILE}")
            return CACHE_FILE

    print(f"Downloading {OSM_URL}...")
    try:
        urlretrieve(OSM_URL, CACHE_FILE)
        print(f"Downloaded to {CACHE_FILE}")
        return CACHE_FILE
    except URLError as e:
        print(f"Failed to download OSM: {e}")
        if CACHE_FILE.exists():
            print("Falling back to cached file")
            return CACHE_FILE
        raise


def extract_astana_bbox(input_path: Path, output_path: Path) -> Path:
    """Extract Astana bounding box using osmium (if available)."""
    import shutil
    import subprocess

    if not shutil.which("osmium"):
        print("osmium not found — skipping bbox extraction, will filter programmatically")
        return input_path

    bbox_str = f"{ASTANA_BBOX['west']},{ASTANA_BBOX['south']},{ASTANA_BBOX['east']},{ASTANA_BBOX['north']}"
    cmd = [
        "osmium", "extract",
        "--bbox", bbox_str,
        "--set-bounds",
        "--overwrite",
        "-o", str(output_path),
        str(input_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


if __name__ == "__main__":
    download_osm()
