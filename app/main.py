from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from celery.result import AsyncResult

from app.core.celery_app import celery_app
from app.utils.task_router import create_task_router
from app.tasks.example_tasks import ask_groq, GroqTaskPayload
from app.tasks.learning_plan_tasks import create_learning_plan, LearningPlanTaskPayload
from app.tasks.learning_plan_tasks_v2 import (
    create_learning_path_v2,
    LearningPathTaskPayloadv2,
)

# Create FastAPI app
app = FastAPI(title="FastAPI Job Queue with Groq", version="1.0")


# --- Job Status Endpoint ---
class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: str | dict | None = None


@app.get(
    "/jobs/{job_id}", response_model=JobStatusResponse, status_code=status.HTTP_200_OK
)
def get_job_status(job_id: str):
    """
    Retrieve the status and result of a background job.
    Maps Celery states to: queued, running, completed, failed.
    """
    task_result = AsyncResult(job_id, app=celery_app)

    status_mapping = {
        "PENDING": "queued",
        "STARTED": "running",
        "SUCCESS": "completed",
        "FAILURE": "failed",
    }
    current_status = status_mapping.get(task_result.status, "unknown")

    result = task_result.result if task_result.ready() else None

    return JobStatusResponse(job_id=job_id, status=current_status, result=result)


# --- Auto-generate and include task routes ---
# For each task, create and include its router.
groq_router = create_task_router(
    task=ask_groq, payload_model=GroqTaskPayload, task_name="ask-groq"
)
app.include_router(groq_router, prefix="/api", tags=["Groq Tasks"])

create_learning_plan_router = create_task_router(
    task=create_learning_plan,
    payload_model=LearningPlanTaskPayload,
    task_name="create-learning-plan",
)
app.include_router(
    create_learning_plan_router, prefix="/api", tags=["Create Learning Plan"]
)

create_learning_path_v2_router = create_task_router(
    task=create_learning_path_v2,
    payload_model=LearningPathTaskPayloadv2,
    task_name="create-learning-path-v2",
)
app.include_router(
    create_learning_path_v2_router, prefix="/api", tags=["Create Learning Path V2"]
)


@app.get("/", summary="Health Check")
def read_root():
    return {"status": "ok"}
