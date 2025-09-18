from celery import Celery
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    REDIS_URL: str
    class Config:
        env_file = '.env'

settings = Settings()

# Configure Celery
# The result_expires setting is in seconds (24 hours)
celery_app = Celery(
    "worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    broker_connection_retry_on_startup=True,
    result_expires=86400,
)

celery_app.conf.update(
    task_track_started=True,
    include=["app.tasks.example_tasks", "app.tasks.learning_plan_tasks", "app.tasks.learning_plan_tasks_v2"],
)