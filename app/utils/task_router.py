from fastapi import APIRouter, status, Body
from pydantic import BaseModel
from celery.result import AsyncResult
from celery.app.task import Task

def create_task_router(task: Task, payload_model: type[BaseModel], task_name: str) -> APIRouter:
    """
    Creates a FastAPI router with synchronous and asynchronous endpoints for a given Celery task.
    """
    router = APIRouter()

    @router.post(f"/sync/{task_name}", status_code=status.HTTP_200_OK)
    def sync_endpoint(payload: payload_model = Body(...)):
        """Direct, blocking execution of the task."""
        # Note: This runs the task's logic in the current process, not in a worker.
        result = task.apply(args=[payload.model_dump()]).get()
        return {"status": "completed", "result": result}

    @router.post(f"/async/{task_name}", status_code=status.HTTP_202_ACCEPTED)
    def async_endpoint(payload: payload_model = Body(...)):
        """Queues the task for background execution."""
        task_result = task.delay(payload.model_dump())
        return {"status": "queued", "job_id": task_result.id}
    
    return router