import os
from pathlib import Path
from pystac_client import Client
import planetary_computer
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

def create_retrying_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

class SentinelIngestor:
    """
    Automated ingestion pipeline for Sentinel-2 L2A optical imagery using 
    Microsoft Planetary Computer.
    """
    def __init__(self, output_dir: str = "data/sentinel2_raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Initialize STAC client
        self.catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        self.session = create_retrying_session()

    def query_items(self, bbox: list, time_range: str, max_cloud_cover: float = 20.0, max_items: int = 1):
        """
        Query the STAC catalog and return matching items.
        bbox Format: [min_lon, min_lat, max_lon, max_lat]
        time_range Format: "2023-01-01/2023-01-31"
        """
        print(f"Querying Sentinel-2 over bbox {bbox} for {time_range}...")
        search = self.catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=time_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}}
        )
        
        items = list(search.get_items())
        print(f"Discovered {len(items)} candidate items.")
        return items[:max_items]

    def query_and_download(self, bbox: list, time_range: str, max_cloud_cover: float = 20.0, max_items: int = 1):
        """
        Query the STAC catalog and download the True Color (B04, B03, B02) GeoTIFFs.
        bbox Format: [min_lon, min_lat, max_lon, max_lat]
        time_range Format: "2023-01-01/2023-01-31"
        """
        items = self.query_items(
            bbox=bbox,
            time_range=time_range,
            max_cloud_cover=max_cloud_cover,
            max_items=max_items,
        )

        downloaded_paths = []
        for item in items:
            item_id = item.id
            cloud_cover = item.properties.get("eo:cloud_cover", 0)
            print(f"Processing Item {item_id} (Cloud Cover: {cloud_cover:.1f}%)")
            
            # The 'visual' asset typically contains the 10m True Color composite (RGB)
            asset = item.assets.get("visual")
            if not asset:
                print(f"Visual asset not found for {item_id}. Skipping.")
                continue
                
            href = asset.href
            target_path = self.output_dir / f"{item_id}_TCI.tif"
            
            if target_path.exists():
                print(f"File {target_path.name} already exists. Skipping download.")
                downloaded_paths.append(target_path)
                continue
                
            self._download_file(href, target_path)
            downloaded_paths.append(target_path)
            
        return downloaded_paths

    def _download_file(self, url: str, target_path: Path):
        print(f"Downloading {url} to {target_path}...")
        response = self.session.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1 Megabyte
        
        with open(target_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=target_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print(f"Successfully downloaded {target_path.name}")

if __name__ == "__main__":
    # Test execution for Dubai Intl Airport
    ingestor = SentinelIngestor()
    # [min_lon, min_lat, max_lon, max_lat]
    dubai_bbox = [55.33, 25.23, 55.40, 25.27]
    
    # Grab an image from early 2023 and one from late 2023
    t1_paths = ingestor.query_and_download(dubai_bbox, "2023-01-01/2023-01-30", max_items=1)
    t2_paths = ingestor.query_and_download(dubai_bbox, "2023-11-01/2023-11-30", max_items=1)
    
    print("\nTest Ingestion Complete.")
    print("T1 Path:", t1_paths)
    print("T2 Path:", t2_paths)
# DEPRECATED: superseded by backend/pipeline/
