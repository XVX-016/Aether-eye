from satellite_ingestion.stac_query import SentinelIngestor
from tiling_engine.tile_generator import TileGenerator
from tiling_engine.spectral_filter import SpectralFilter
from pathlib import Path

def test_pipeline():
    print("--- Stage 1: Global Ingestion ---")
    ingestor = SentinelIngestor(output_dir="data/sentinel_test")
    # Tiny BBox over Dubai Airport
    bbox = [55.33, 25.23, 55.40, 25.27]
    
    # Expand dates and allow up to 30% cloud cover to guarantee a match
    t1_tiffs = ingestor.query_and_download(bbox, "2023-01-01/2023-03-31", max_cloud_cover=30.0)
    t2_tiffs = ingestor.query_and_download(bbox, "2023-10-01/2023-12-31", max_cloud_cover=30.0)
    
    if not t1_tiffs or not t2_tiffs:
        print("Failed to download T1 and T2 pairs from Planetary Computer. Check cloud cover / dates.")
        return

    print("\n--- Stage 2: Tiling Engine ---")
    tiler = TileGenerator(tile_size=512)
    
    # We expect 1 TIF each since max_items=1 is the default
    t1_tif = t1_tiffs[0]
    t2_tif = t2_tiffs[0]
    
    t1_output = Path("data/tiles_t1")
    t2_output = Path("data/tiles_t2")
    
    t1_tiles = tiler.generate_tiles(t1_tif, t1_output)
    t2_tiles = tiler.generate_tiles(t2_tif, t2_output)
    
    print("\n--- Stage 3: Cheap Mathematical Filter ---")
    filter_engine = SpectralFilter(diff_threshold=20.0, min_changed_pixels=100)
    suspicious_pairs = filter_engine.filter_tiles(t1_output, t2_output)
    
    print("\n--- Pipeline Summary ---")
    print(f"Original Tiles Generated: {len(t1_tiles)}")
    print(f"Empty Tiles Dropped:      {len(t1_tiles) - len(suspicious_pairs)}")
    print(f"Suspicious Tiles Forwarded to Siamese_Unet: {len(suspicious_pairs)}")

if __name__ == "__main__":
    test_pipeline()
