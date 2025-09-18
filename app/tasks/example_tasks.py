import os
from groq import Groq
from pydantic import BaseModel, Field
from app.core.celery_app import celery_app

# Pydantic model for the task payload
class GroqTaskPayload(BaseModel):
    prompt: str = Field(..., example="Explain the importance of low-latency LLMs")
    model: str = Field("llama3-8b-8192", example="llama3-8b-8192")

@celery_app.task
def ask_groq(payload: dict):
    """
    A Celery task to interact with the Groq API.
    The payload is a dictionary representation of GroqTaskPayload.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": payload['prompt']}],
        model=payload['model'],
    )
    
    return chat_completion.choices[0].message.content

@celery_app.task
def ask_groq_2(payload: dict):
    """
    A Celery task to interact with the Groq API.
    The payload is a dictionary representation of GroqTaskPayload.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": payload['prompt']}],
        model=payload['model'],
        temperature=0.7,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )
    
    return chat_completion.choices[0].message.content