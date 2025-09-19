import os
import json
from groq import Groq
from pydantic import BaseModel, Field, ValidationError
from app.core.celery_app import celery_app
import httpx
from typing import List, Literal, Optional

# --- Subject ID Mapping ---
# Maps the subject names identified by the AI to your specific MongoDB ObjectIDs.
SUBJECT_ID_MAP = {
    "Physics": "68c97d193c5fb93d44667a6c",
    "Chemistry": "68cbc78bcd2937d0d4dda433",
    "Mathematics": "68cbc7abcd2937d0d4dda434",
}

# --- System Prompts ---

# 1. Main prompt for generating the core MCQ content.
SYSTEM_PROMPT_GENERATE_MCQ = """
You are an expert question creator and academic tutor specializing in the Indian competitive examination syllabus for IIT-JEE (Mains and Advanced) and NEET. Your sole purpose is to generate a single, high-quality, original multiple-choice question (MCQ) based on a user's topic query.

**Your instructions are absolute:**

1.  **Analyze the Query:** Carefully parse the user's query to identify the subject (Physics, Chemistry, Maths, Biology), the specific topic, and the target examination level (JEE Mains, JEE Advanced, or NEET).
2.  **Default Level:** If the examination level is not specified or is ambiguous, you MUST default to **JEE Mains**.
3.  **Question Quality:** The question must be conceptually sound, challenging, and directly relevant to the specified syllabus. It should not be a simple definition recall but should test application, analysis, or problem-solving skills appropriate for the target level.
4.  **Strict Output Format:** You MUST reply with ONLY a single, raw JSON object. Do not include any introductory text, explanations, or markdown formatting like ```json. Your entire response must be the JSON object itself.
5.  **JSON Schema:** The JSON object must strictly adhere to the following structure:
    {
      "question": "The full text of the question, including any necessary values or conditions.",
      "options": {
        "A": "Option A text.",
        "B": "Option B text.",
        "C": "Option C text.",
        "D": "Option D text."
      },
      "correct_answer": "The key of the correct option (e.g., 'C').",
      "explanation": "A detailed, step-by-step explanation that thoroughly derives the correct answer and, if applicable, explains why the other options are incorrect. This should be comprehensive enough for a student to learn from."
    }
"""

# 2. Secondary prompt for extracting metadata from the user's query.
SYSTEM_PROMPT_EXTRACT_METADATA = """
You are an AI assistant that analyzes a user's query for creating an exam question and extracts structured metadata.

**Your instructions are absolute:**

1.  **Analyze the Query:** Read the user's query carefully.
2.  **Identify Subject:** Determine the primary subject from the query. It must be one of: "Physics", "Chemistry", "Mathematics".
3.  **Identify Difficulty:** Determine the difficulty level. Map "JEE Mains" or "NEET" to "medium". Map "JEE Advanced" to "hard". If no level is specified, default to "medium".
4.  **Extract Tags:** Identify 2-4 key technical terms from the query to use as search tags.
5.  **Strict Output Format:** Respond with ONLY a single, raw JSON object with the following schema:
    {
        "subject": "The identified subject (e.g., 'Physics')",
        "difficulty": "The identified difficulty ('easy', 'medium', or 'hard')",
        "tags": ["tag1", "tag2"]
    }
"""


# --- Pydantic Models ---


# Pydantic model for the incoming task payload
class QuestionGenerationPayload(BaseModel):
    query: str = Field(
        ..., example="JEE advanced rotational mechanics rolling without slipping"
    )


# Pydantic models for validating the LLM's MCQ output
class MCQOptions(BaseModel):
    A: str
    B: str
    C: str
    D: str


class MCQQuestion(BaseModel):
    question: str
    options: MCQOptions
    correct_answer: str = Field(..., pattern="^[A-D]$")
    explanation: str


# Pydantic models for validating the LLM's metadata output
class ExtractedMetadata(BaseModel):
    subject: str
    difficulty: Literal["easy", "medium", "hard"]
    tags: List[str]


# --- Celery Task Definition ---


@celery_app.task(bind=True, max_retries=3, default_retry_delay=15)
def generate_question(self, payload: dict):
    """
    Generates a structured MCQ and formats it for the IQuestion Mongoose schema.

    This task performs two main steps:
    1.  Generates the core question content (text, options, explanation).
    2.  Extracts metadata (subject, difficulty, tags) from the user query.
    3.  Transforms the combined data into the target database format.
    """
    # Initialize response_content to None to handle potential errors before assignment
    response_content = None
    try:
        # 1. Validate the input payload
        validated_payload = QuestionGenerationPayload.model_validate(payload)
        user_query = validated_payload.query
        print(f"✅ Received valid request to generate question for: '{user_query}'")

        # 2. Initialize the Groq client
        client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

        # --- Step A: Generate the core MCQ content ---
        print("📞 Calling Groq API for MCQ generation...")
        mcq_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_GENERATE_MCQ},
                {"role": "user", "content": user_query},
            ],
            model="openai/gpt-oss-120b",
            temperature=0.5,
            response_format={"type": "json_object"},
        )
        response_content = mcq_completion.choices[0].message.content
        mcq_data = json.loads(response_content)
        validated_mcq = MCQQuestion.model_validate(mcq_data)
        print("👍 Successfully generated and validated the MCQ content.")

        # --- Step B: Extract metadata from the user query ---
        print("📞 Calling Groq API for metadata extraction...")
        metadata_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_EXTRACT_METADATA},
                {"role": "user", "content": user_query},
            ],
            model="openai/gpt-oss-120b",  # Can use a faster model if available
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        metadata_content = metadata_completion.choices[0].message.content
        metadata_data = json.loads(metadata_content)
        validated_metadata = ExtractedMetadata.model_validate(metadata_data)
        print(f"👍 Successfully extracted metadata: {validated_metadata.model_dump()}")

        # --- Step C: Transform the data to match the IQuestion schema ---
        print("🔄 Transforming data into the target schema...")

        # Transform options from {"A": "text", ...} to [{text: "...", isCorrect: ...}]
        transformed_options = [
            {"text": text, "isCorrect": key == validated_mcq.correct_answer}
            for key, text in validated_mcq.options.model_dump().items()
        ]

        # Map the subject name to its MongoDB ObjectID
        subject_id = SUBJECT_ID_MAP.get(validated_metadata.subject)
        if not subject_id:
            print(
                f"⚠️ Warning: Could not map subject '{validated_metadata.subject}'. Defaulting to None."
            )

        # Assemble the final document for the database
        # NOTE: The 'explanation' is not part of the IQuestion schema provided.
        # We are returning it alongside the main document so it can be used elsewhere if needed.
        final_document = {
            "text": validated_mcq.question,
            "imageUrl": None,  # Not generated in this process
            "options": transformed_options,
            "difficulty": validated_metadata.difficulty,
            "subject": subject_id,
            "tags": validated_metadata.tags,
        }

        # The task returns a dict containing the formatted question and the explanation
        task_output = {
            "question_data": final_document,
            "explanation": validated_mcq.explanation,
        }

        print("✅ Successfully formatted data for IQuestion model.")
        return task_output

    except (ValidationError, json.JSONDecodeError) as e:
        print(
            f"⚠️ Validation Error: The LLM response was not in the expected format. Error: {e}"
        )
        print(f"   Raw response was: {response_content}")
        raise self.retry(exc=e)

    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        raise self.retry(exc=e)
