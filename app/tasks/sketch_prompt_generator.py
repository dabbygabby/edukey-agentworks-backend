# app/tasks/sketch_prompt_generator.py

import os
import json
from groq import Groq
from pydantic import BaseModel, Field, ValidationError
from app.core.celery_app import celery_app
import httpx

# --- System Prompt ---
# This prompt instructs the LLM to read a physics question and its explanation
# and distill it into a simple, one-sentence description of the physical setup.

SYSTEM_PROMPT = """
You are an AI assistant that specializes in summarizing physics problems for visual representation. You will be given the text of a multiple-choice question and its detailed explanation.

Your task is to read the problem and generate a single, concise, one-sentence description of the physical setup. This description will be used as a prompt for an image generation model.

**CRITICAL INSTRUCTIONS:**
1.  **Focus ONLY on the initial physical arrangement of objects.** Describe the scene before any action happens.
2.  **DO NOT describe the question being asked or the solution.**
3.  **DO NOT include specific values, numbers, or complex variables.** Keep it generic (e.g., "a block", "an angle theta").
4.  **Your output MUST BE ONLY the descriptive sentence and nothing else.** Do not add any introductory phrases like "Here is the description:".

**EXAMPLES:**
- **Input:** A question about a block of mass 5kg sliding down a 30-degree ramp.
  **Output:** A block on an inclined plane.
- **Input:** A question about a pendulum of length L swinging.
  **Output:** A simple pendulum hanging from a pivot point.
- **Input:** A question about two masses connected by a pulley on a table.
  **Output:** Two masses connected by a string over a pulley, with one mass on a table.
"""

# --- Pydantic Models ---


# Defines the expected input payload for this task
class SketchPromptGenerationPayload(BaseModel):
    question_text: str = Field(
        ..., example="A uniform thin rod of length L and mass M is hinged..."
    )
    explanation_text: str = Field(
        ..., example="When the rod rotates from the horizontal..."
    )


# --- Celery Task Definition ---


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def generate_sketch_prompt(self, payload: dict):
    """
    Takes a generated question and explanation, and creates a concise prompt
    for the sketch generation task.
    """
    response_content = None
    try:
        # 1. Validate the input payload
        validated_payload = SketchPromptGenerationPayload.model_validate(payload)
        print("✅ Received valid request to generate sketch prompt.")

        # 2. Combine the texts to form a rich context for the LLM
        full_context = f"QUESTION: {validated_payload.question_text}\n\nEXPLANATION: {validated_payload.explanation_text}"

        # 3. Initialize the Groq client
        client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

        # 4. Make the API call to Groq to generate the summary prompt
        print("📞 Calling Groq API to summarize for sketch prompt...")
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_context},
            ],
            model="openai/gpt-oss-120b",  # A powerful model is good for summarization
            temperature=0.1,  # Low temperature for factual, direct summarization
        )

        description = chat_completion.choices[0].message.content
        print(f"👍 Successfully generated sketch prompt: '{description}'")

        # 5. Return the result in a structured format
        return {"status": "completed", "description": description.strip()}

    except ValidationError as e:
        print(f"⚠️ Input Validation Error: {e}")
        return {"status": "failed", "error": f"Invalid input payload: {e}"}

    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        # Retry the task for transient errors
        raise self.retry(exc=e)
