# app/tasks/sketch_generator.py

import os
import uuid
import traceback
from groq import Groq
from pydantic import BaseModel, Field
from app.core.celery_app import celery_app
import httpx
from vercel_blob import put  # Import the put function

# --- System Prompt for Matplotlib ---
# This prompt remains unchanged.
SYSTEM_PROMPT = """
You are an expert Python programmer specializing in the Matplotlib library for creating scientific figures. Your task is to convert a user's natural language description of a physics problem into a complete, executable Python script that generates and saves a PNG image of the sketch.

**MATPLOTLIB USAGE PATTERN:**
1.  **Import:** Start with `import matplotlib.pyplot as plt` and `import numpy as np`. You might also need `import matplotlib.patches as patches`.
2.  **Figure and Axes:** Create a figure and axes object: `fig, ax = plt.subplots(figsize=(8, 8))`.
3.  **Drawing Shapes:**
    * **Line:** `ax.plot([x1, x2], [y1, y2], color='k', linewidth=2)`
    * **Circle/Bob:** `circle = plt.Circle((x, y), radius, color='b', zorder=5)` followed by `ax.add_patch(circle)`.
    * **Arc:** `arc = patches.Arc((x, y), width, height, angle=0, theta1=start_deg, theta2=end_deg, color='r')` followed by `ax.add_patch(arc)`.
    * **Text/Labels:** `ax.text(x, y, r'$\\theta$', fontsize=15, ha='center', va='center')`. Use LaTeX for math symbols.
4.  **Appearance:**
    * Set axis limits: `ax.set_xlim(min, max)` and `ax.set_ylim(min, max)`.
    * Ensure correct proportions: `ax.set_aspect('equal', adjustable='box')`.
    * Turn off axes for a clean sketch: `ax.axis('off')`.
5.  **Saving:** The final line must be `plt.savefig('filename.png', bbox_inches='tight', dpi=150)`.

**ABSOLUTE RULES:**
1.  **CODE ONLY:** Your entire response MUST be raw Python code. Do not use markdown or explanations.
2.  **MATPLOTLIB ONLY:** You must use the `matplotlib.pyplot` library. Do NOT use `pysketcher`.
3.  **IMPORTS:** The script must start with the necessary imports.
4.  **WORKFLOW:** Follow the Matplotlib usage pattern precisely.
5.  **SAVE THE FILE:** The script's final, non-comment line MUST be `plt.savefig('...')` with the exact filename provided.
"""

# --- Pydantic Model for Task Input ---


class SketchTaskPayload(BaseModel):
    description: str = Field(
        ...,
        example="A block of mass m on a frictionless inclined plane with an angle alpha.",
    )


# --- Celery Task Definition (Updated for Vercel Blob) ---


@celery_app.task
def generate_sketch(payload: dict):
    """
    A Celery task to generate a physics sketch and upload it to Vercel Blob.
    1. Sends a description to a Groq LLM to generate Matplotlib code.
    2. Executes the code to save a temporary local PNG file.
    3. Reads the local file and uploads its contents to Vercel Blob.
    4. Deletes the temporary local file.
    5. Returns the public URL of the sketch.
    """
    python_code = ""
    filename = ""  # Initialize filename to be accessible in finally block
    try:
        # 1. Validate the input payload
        validated_payload = SketchTaskPayload.model_validate(payload)
        user_description = validated_payload.description
        print(f"✅ Received request to generate sketch for: '{user_description}'")

        # 2. Prepare for file generation
        output_dir = "sketches_temp"  # Use a temporary local directory
        os.makedirs(output_dir, exist_ok=True)
        unique_id = uuid.uuid4()
        filename = f"{output_dir}/sketch_{unique_id}.png"
        blob_pathname = f"sketches/sketch_{unique_id}.png"  # Path in blob storage

        # 3. Initialize the Groq client
        client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=httpx.Timeout(180.0, connect=10.0),
        )

        # 4. Construct the user prompt for the LLM
        user_prompt = f"""
Generate a Python script using Matplotlib to draw: '{user_description}'.
The final line of the script MUST be exactly: plt.savefig('{filename}', bbox_inches='tight', dpi=150)
"""

        # 5. Make the API call to Groq
        print("📞 Calling Groq API to generate Matplotlib code...")
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model="openai/gpt-oss-120b",
            temperature=0.2,
        )

        python_code = chat_completion.choices[0].message.content
        print("🐍 Received Matplotlib code from Groq.")
        print("--- Generated Code ---\n" + python_code + "\n----------------------")

        # 6. Execute the generated Matplotlib code
        print(f"⚡ Executing code to generate temporary file: {filename}...")
        exec(python_code)

        # 7. Verify local file, upload to Vercel Blob, and return URL
        if os.path.exists(filename):
            print(f"👍 Successfully created temporary local sketch: {filename}")

            # Read the file's binary content
            with open(filename, "rb") as f:
                file_data = f.read()

            # Upload to Vercel Blob
            print(f"☁️ Uploading to Vercel Blob as '{blob_pathname}'...")

            # 👇 FIX: Pass 'access' inside the 'options' dictionary
            blob_result = put(blob_pathname, file_data, options={"access": "public"})

            print(f"✅ Upload complete! URL: {blob_result['url']}")

            # Return the public URL
            return {"status": "completed", "url": blob_result["url"]}
        else:
            print(f"❌ Execution finished, but file '{filename}' was not created.")
            return {
                "status": "failed",
                "error": "Code executed, but the output file was not saved locally.",
            }

    except Exception as e:
        print(f"❌ An error occurred during sketch generation: {repr(e)}")
        traceback.print_exc()
        print(f"   Failed code was:\n{python_code}")
        return {"status": "failed", "error": str(e)}

    finally:
        # Always clean up the local temporary file if it was created
        if filename and os.path.exists(filename):
            os.remove(filename)
            print(f"🧹 Cleaned up temporary file: {filename}")
