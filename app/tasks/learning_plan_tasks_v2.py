import os
import json
from groq import Groq
from pydantic import BaseModel, Field
from app.core.celery_app import celery_app
import httpx

# --- Pydantic Models ---


class LearningPathTaskPayloadv2(BaseModel):
    """Defines the input payload for the learning path creation task."""

    topic: str = Field(..., example="Rotational Motion")
    detail_level: str = Field(
        "advanced", example="advanced", description="Can be 'mains' or 'advanced'"
    )


# --- Celery Task: Learning Path Creator V2 (Robust Version) ---


@celery_app.task(name="tasks.create_learning_path_v2")
def create_learning_path_v2(payload: dict):
    """
    Orchestrates a robust, multi-step process using Groq to generate a detailed,
    hierarchical learning path for IIT-JEE aspirants. This version validates LLM
    outputs at each step to prevent crashes from malformed JSON.
    """
    try:
        # 1. Unpack and validate the input payload
        topic = payload.get("topic")
        detail_level = payload.get("detail_level", "advanced")
        if not topic:
            raise ValueError("Payload must include a 'topic' key.")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")

        # Initialize Groq client with a generous timeout
        client = Groq(api_key=api_key, timeout=httpx.Timeout(300.0, connect=10.0))
        print(
            f'✅ [V2-Robust] Starting learning path generation for topic: "{topic}" (Level: {detail_level})'
        )

        # --- STEP 1: Search for authoritative URLs ---
        print(
            "\n--- STEP 1: Searching for authoritative sources with groq/compound... ---"
        )
        search_prompt = (
            f'Perform a web search to find 3-4 highly authoritative URLs for learning about "{topic}" '
            f"for the Indian IIT-JEE {detail_level} syllabus. Focus on academic or reputable educational sites. "
            'Return ONLY a single JSON object with a key "urls" containing an array of the found URLs.'
        )
        urls = []
        try:
            search_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": search_prompt}],
                model="groq/compound",
                response_format={"type": "json_object"},
            )
            search_result_content = search_completion.choices[0].message.content
            if search_result_content:
                urls = json.loads(search_result_content).get("urls", [])
        except Exception as e:
            print(f"   -> ❌ Error during web search step: {repr(e)}")

        print(f"   -> Found {len(urls)} sources: {urls if urls else 'None'}")

        # --- STEP 2: Generate High-Level Structure ---
        print("\n--- STEP 2: Generating the high-level learning path structure... ---")
        structure_prompt = f"""
        Act as an expert academic curriculum designer for IIT-JEE coaching in India.
        For the topic "{topic}" targeting the "{detail_level}" level, create a hierarchical learning path structure.
        Generate a pure JSON object with a single root key representing the chapter name. This key should contain a list of "topics".
        Each "topic" object must have: "topic_name" (string), "prerequisites" (empty list), "problem_solving_tips" (empty list), "common_pitfalls" (empty list), and "concepts" (a list of "concept" objects).
        Each "concept" object must have: "concept_name" (string), "reading_material" (empty string), and "mcqs" (empty list).
        Break down "{topic}" into logical topics and concepts essential for the JEE syllabus. Do NOT populate the empty fields yet.
        """
        learning_path = {}
        try:
            structure_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": structure_prompt}],
                model="openai/gpt-oss-120b",
                response_format={"type": "json_object"},
            )
            learning_path = json.loads(structure_completion.choices[0].message.content)
            print("   -> Successfully generated empty learning path skeleton.")
        except Exception as e:
            print(
                f"   -> ❌ Critical Error: Failed to generate skeleton. Aborting. Error: {repr(e)}"
            )
            return None  # Cannot proceed without the skeleton

        # --- STEP 3: Build Knowledge Base from Sources ---
        print("\n--- STEP 3: Building knowledge base from sources... ---")
        knowledge_base = ""
        if not urls:
            print("   -> ⚠️ No URLs found, will rely on model's internal knowledge.")
        else:
            for i, url in enumerate(urls):
                print(f"   -> [{i+1}/{len(urls)}] Visiting and summarizing: {url}")
                try:
                    visit_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": f"Visit and provide a detailed summary of the key academic points from this page, focusing on formulas, definitions, and core principles relevant to IIT-JEE Physics/Chemistry/Maths: {url}",
                            }
                        ],
                        model="groq/compound",
                    )
                    summary = (
                        visit_completion.choices[0].message.content
                        or "No summary available."
                    )
                    knowledge_base += f"--- Source from {url} ---\n{summary}\n\n"
                    print(f"   -> Summary acquired for {url}")
                except Exception as e:
                    print(f"   -> ❌ Failed to visit or summarize URL {url}: {e}")
            print(
                f"   -> Knowledge base built. Total length: {len(knowledge_base)} characters."
            )

        # --- STEP 4: Iteratively Populate the Structure ---
        print("\n--- STEP 4: Populating the structure with detailed content... ---")
        if (
            not learning_path
            or not isinstance(learning_path, dict)
            or len(learning_path.keys()) == 0
        ):
            print(
                "   -> ❌ Critical Error: Learning path skeleton is invalid. Aborting."
            )
            return None

        chapter_key = list(learning_path.keys())[0]

        for topic_obj in learning_path[chapter_key]:
            topic_name = topic_obj.get("topic_name", "Unknown Topic")
            print(f"\n  -> Generating details for Topic: '{topic_name}'")

            # 4a: Populate topic details (prerequisites, tips, pitfalls)
            topic_details_prompt = f"""
            Based on the context below, generate details for the IIT-JEE topic "{topic_name}" within the chapter "{topic}".
            CONTEXT: {knowledge_base if knowledge_base else "No web context available. Rely on your internal knowledge."}
            ---
            Generate a single JSON object with keys: "prerequisites", "problem_solving_tips", "common_pitfalls".
            """
            try:
                topic_details_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": topic_details_prompt}],
                    model="openai/gpt-oss-120b",
                    response_format={"type": "json_object"},
                )
                topic_details = json.loads(
                    topic_details_completion.choices[0].message.content
                )

                # *** VALIDATION STEP ***
                if isinstance(topic_details, dict):
                    topic_obj.update(topic_details)
                    print(f"    -> Details for topic '{topic_name}' populated.")
                else:
                    print(
                        f"    -> ⚠️ Warning: Received invalid data type for topic details for '{topic_name}'. Skipping update."
                    )

            except Exception as e:
                print(
                    f"    -> ❌ Error generating details for topic '{topic_name}': {repr(e)}"
                )

            # 4b: Populate content for each concept
            if "concepts" in topic_obj and isinstance(topic_obj["concepts"], list):
                for concept_obj in topic_obj["concepts"]:
                    concept_name = concept_obj.get("concept_name", "Unknown Concept")
                    print(f"    -> Generating content for Concept: '{concept_name}'")

                    concept_content_prompt = f"""
                    Act as a master teacher for IIT-JEE. Use the context below to generate content for the concept "{concept_name}" under the topic "{topic_name}".
                    CONTEXT: {knowledge_base if knowledge_base else "No web context available. Rely on your internal knowledge."}
                    ---
                    Generate a single JSON object with two keys: "reading_material" (string) and "mcqs" (an array of MCQ objects).
                    Each MCQ object must have: "question", "options" (array), "correct_answer_index" (integer), and "explanation".
                    """
                    try:
                        concept_content_completion = client.chat.completions.create(
                            messages=[
                                {"role": "user", "content": concept_content_prompt}
                            ],
                            model="openai/gpt-oss-120b",
                            response_format={"type": "json_object"},
                        )
                        concept_content = json.loads(
                            concept_content_completion.choices[0].message.content
                        )

                        # *** VALIDATION STEP ***
                        if (
                            isinstance(concept_content, dict)
                            and "reading_material" in concept_content
                            and "mcqs" in concept_content
                        ):
                            concept_obj.update(concept_content)
                        else:
                            print(
                                f"    -> ⚠️ Warning: Received malformed content for concept '{concept_name}'. Skipping update."
                            )

                    except Exception as e:
                        print(
                            f"    -> ❌ Error generating content for concept '{concept_name}': {repr(e)}"
                        )

        # --- STEP 5: Finalize and Reformat Output ---
        print(
            "\n\n--- ✅ Learning Path Synthesis Complete! Formatting final output... ---"
        )
        final_output = {"topic": chapter_key, "content": learning_path[chapter_key]}
        return final_output

    except Exception as e:
        print(
            f"\n--- ❌ A top-level error occurred during learning path generation ---"
        )
        print(f"Error: {repr(e)}")
        return None
