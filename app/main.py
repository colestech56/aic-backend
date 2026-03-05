"""
AIC Backend — Adaptive Intervention Protocol API.

FastAPI application with Thompson sampling, Bayesian shrinkage,
LLM-generated interventions, and survey scheduling.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.db.database import engine, Base, async_session
from app.routers import participants, surveys, interventions, boost, admin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Fixed UUIDs matching the frontend hardcoded IDs
DEMO_PARTICIPANT_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
DEMO_SCHEDULE_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


async def seed_demo_data():
    """Create a demo participant and survey schedule if they don't exist."""
    from app.models.participant import Participant, ParticipantPreference
    from app.models.bandit import BanditArm
    from app.models.threshold import ThresholdState
    from app.models.survey import SurveySchedule
    from app.services.thompson_sampling import create_fresh_arms
    from sqlalchemy import select

    async with async_session() as db:
        # Check if demo participant exists
        result = await db.execute(
            select(Participant).where(Participant.id == DEMO_PARTICIPANT_ID)
        )
        if result.scalar_one_or_none():
            # Already seeded — refresh the survey schedule window so it stays open
            sched_result = await db.execute(
                select(SurveySchedule).where(SurveySchedule.id == DEMO_SCHEDULE_ID)
            )
            schedule = sched_result.scalar_one_or_none()
            if schedule:
                schedule.status = "scheduled"
                schedule.completed_at = None
                schedule.window_closes_at = datetime.utcnow() + timedelta(hours=24)
                await db.commit()
            logger.info("Demo data exists, refreshed survey window")
            return

        # Create demo participant
        participant = Participant(
            id=DEMO_PARTICIPANT_ID,
            study_id="AIC-DEMO-001",
            diagnostic_group="SZ",
            condition="emi",
        )
        db.add(participant)

        # Preferences
        prefs = ParticipantPreference(
            participant_id=DEMO_PARTICIPANT_ID,
            preferred_activities=["walking", "listening to music", "stretching"],
        )
        db.add(prefs)

        # Initialize all bandit arms with Beta(1,1) priors
        for bandit_type in ["timing", "intervention_type", "boost"]:
            arms = create_fresh_arms(bandit_type)
            for arm in arms:
                db.add(BanditArm(
                    participant_id=DEMO_PARTICIPANT_ID,
                    bandit_type=bandit_type,
                    arm_name=arm.arm_name,
                    arm_index=arm.arm_index,
                    alpha=1.0,
                    beta_param=1.0,
                ))

        # Threshold state
        db.add(ThresholdState(
            participant_id=DEMO_PARTICIPANT_ID,
            current_threshold=4.0,
        ))

        # Survey schedule — always open (rolling 24h window, reset each startup)
        db.add(SurveySchedule(
            id=DEMO_SCHEDULE_ID,
            participant_id=DEMO_PARTICIPANT_ID,
            survey_type="ema",
            scheduled_at=datetime.utcnow(),
            window_closes_at=datetime.utcnow() + timedelta(hours=24),
            status="scheduled",
        ))

        await db.commit()
        logger.info("Demo data seeded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    logger.info("AIC backend starting up (env=%s)", settings.ENVIRONMENT)

    # Import all models so Base.metadata knows about them
    import app.models  # noqa: F401

    # Create all tables automatically
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified")

    # Seed demo data
    await seed_demo_data()

    yield
    logger.info("AIC backend shutting down")


app = FastAPI(
    title="AIC — Adaptive Intervention Protocol",
    description="Thompson sampling-based adaptive ecological momentary intervention engine",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all origins (needed for Netlify → Render cross-origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(participants.router)
app.include_router(surveys.router)
app.include_router(interventions.router)
app.include_router(boost.router)
app.include_router(admin.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions so they don't leak stack traces to clients."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}
