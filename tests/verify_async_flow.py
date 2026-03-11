import asyncio
import httpx
import time

async def test_async_workflow():
    base_url = "http://localhost:8000/api/intelligence"
    
    # 1. Start processing job
    payload = {
        "image_before": "data/processed/change_detection/test/A/test_1.png",
        "image_after": "data/processed/change_detection/test/B/test_1.png"
    }
    
    print("Starting intelligence job...")
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{base_url}/process", params=payload)
        if resp.status_code != 200:
            print(f"Failed to start job: {resp.text}")
            return
        
        job_id = resp.json()["job_id"]
        print(f"Job started. ID: {job_id}")
        
        # 2. Poll status
        while True:
            resp = await client.get(f"{base_url}/status/{job_id}")
            status_data = resp.json()
            status = status_data["status"]
            progress = status_data["progress"]
            print(f"Status: {status} ({progress*100:.0f}%)")
            
            if status in ["completed", "failed"]:
                break
            await asyncio.sleep(2)
            
        if status == "completed":
            print("Job completed successfully!")
            # 3. Verify events in DB
            resp = await client.get(f"{base_url}/events")
            events = resp.json()
            print(f"Retrieved {len(events)} persistent events from DB.")
        else:
            print(f"Job failed: {status_data.get('error')}")

if __name__ == "__main__":
    # Note: This requires the backend to be running
    # I'll just run the core logic locally in a script for verification since I might not want to start the full server
    pass
