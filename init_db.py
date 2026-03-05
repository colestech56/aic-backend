import asyncio
import uuid
from datetime import datetime, timedelta

from app.db.database import engine, Base, async_session
from app.models import Participant, ParticipantPreference, SurveySchedule

async def init_db():
    print("Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print("Seeding demo data...")
    async with async_session() as db:
        pid = uuid.UUID("11111111-1111-1111-1111-111111111111")
        sid = uuid.UUID("22222222-2222-2222-2222-222222222222")
        
        p = await db.get(Participant, pid)
        if not p:
            p = Participant(
                id=pid,
                study_id="DEMO-001",
                diagnostic_group="sz",
                condition="emi",
                active=True,
                timezone="America/New_York"
            )
            db.add(p)
            
            pref = ParticipantPreference(
                participant_id=pid,
                preferred_activities=["listening to music", "going for a walk"]
            )
            db.add(pref)
            
            sched = SurveySchedule(
                id=sid,
                participant_id=pid,
                survey_type="ema",
                status="scheduled",
                scheduled_at=datetime.utcnow(),
                window_closes_at=datetime.utcnow() + timedelta(days=365)
            )
            db.add(sched)
            
            await db.commit()
            print("Demo participant and active survey schedule created!")
        else:
            print("Demo participant already exists.")

if __name__ == "__main__":
    asyncio.run(init_db())
