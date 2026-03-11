import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from app.database.session import init_db, async_session
from app.core.tasks import create_job, process_satellite_intelligence_task, get_job
from sqlalchemy import select
from app.database.models import IntelligenceEvent

async def verify_local():
    print("Initializing test database...")
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_aether.db"
    await init_db()
    
    job_id = create_job()
    print(f"Starting async task for job {job_id}...")
    
    # Run the task directly (simulating what BackgroundTasks does)
    await process_satellite_intelligence_task(job_id, "before.png", "after.png")
    
    job = get_job(job_id)
    print(f"Job final status: {job.status}")
    
    async with async_session() as session:
        result = await session.execute(select(IntelligenceEvent))
        events = result.scalars().all()
        print(f"Events saved to DB: {len(events)}")
        for e in events:
            print(f" - {e.type} at ({e.lat}, {e.lon})")

if __name__ == "__main__":
    asyncio.run(verify_local())
