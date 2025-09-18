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


# --- Celery Task: Learning Path Creator V2 ---


@celery_app.task(name="tasks.create_learning_path_v2")
def create_learning_path_v2(payload: dict):
    """
    Orchestrates a multi-step process using Groq to generate a detailed, hierarchical
    learning path for IIT-JEE aspirants.
    This version processes web content sequentially to avoid context length errors.
    """
    try:
        # Unpack the topic and detail_level from the payload dictionary
        topic = payload.get("topic")
        detail_level = payload.get("detail_level", "advanced")
        if not topic:
            raise ValueError("Payload must include a 'topic' key.")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")

        client = Groq(api_key=api_key, timeout=httpx.Timeout(300.0, connect=10.0))
        print(
            f'✅ [V2] Starting learning path generation for topic: "{topic}" (Level: {detail_level})'
        )

        # --- STEP 1: Search for authoritative URLs using groq/compound ---
        print(
            "\n--- STEP 1: Searching for authoritative sources with groq/compound... ---"
        )
        search_prompt = (
            f'Perform a web search to find 3-4 highly authoritative URLs for learning about "{topic}" '
            f"for the Indian IIT-JEE {detail_level} syllabus. Focus on academic or reputable educational sites. "
            'Return ONLY a single JSON object with a key "urls" containing an array of the found URLs.'
        )

        search_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": search_prompt}],
            model="groq/compound",  # Using the compound model for web search
            response_format={
                "type": "json_object"
            },  # Enforce JSON output for reliable parsing
        )

        urls = []
        search_result_content = search_completion.choices[0].message.content
        if search_result_content:
            try:
                urls = json.loads(search_result_content).get("urls", [])
            except json.JSONDecodeError:
                print(
                    f"   -> ❌ Error: Failed to parse JSON from web search. Raw response: {search_result_content}"
                )
        else:
            print("   -> ❌ Error: Web search returned no content.")

        print(f"   -> Found {len(urls)} sources: {urls if urls else 'None'}")

        # --- STEP 2: Generate High-Level Structure ---
        print("\n--- STEP 2: Generating the high-level learning path structure... ---")
        # (The rest of the code from here remains the same as the previous version)
        structure_prompt = f"""
        Act as an expert academic curriculum designer for IIT-JEE coaching in India.
        For the topic "{topic}" targeting the "{detail_level}" level, create a hierarchical learning path structure.

        Generate a pure JSON object with a single root key representing the chapter name. This key should contain a list of "topics".
        Each "topic" object must have: "topic_name" (string), "prerequisites" (empty list), "problem_solving_tips" (empty list), "common_pitfalls" (empty list), and "concepts" (a list of "concept" objects).
        Each "concept" object must have: "concept_name" (string), "reading_material" (empty string), and "mcqs" (empty list).

        Break down "{topic}" into logical topics and concepts essential for the JEE syllabus. Do NOT populate the empty fields yet.
        """
        structure_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": structure_prompt}],
            model="openai/gpt-oss-120b",
            response_format={"type": "json_object"},
        )
        learning_path = json.loads(structure_completion.choices[0].message.content)
        print("   -> Successfully generated empty learning path skeleton.")

        # --- STEP 3: Build Knowledge Base by Visiting URLs Sequentially ---
        print("\n--- STEP 3: Building knowledge base from sources... ---")
        knowledge_base = ""
        if not urls:
            print("   -> ⚠️ No URLs found, skipping web content summarization.")
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

        # --- STEP 4: Iteratively Populate the Structure with Detailed Content ---
        print("\n--- STEP 4: Populating the structure with detailed content... ---")
        chapter_key = list(learning_path.keys())[0]

        for topic_obj in learning_path[chapter_key]:
            topic_name = topic_obj["topic_name"]
            print(f"\n  -> Generating details for Topic: '{topic_name}'")

            # 4a: Populate prerequisites, tips, and pitfalls
            topic_details_prompt = f"""
            Based on the context below, generate details for the IIT-JEE topic "{topic_name}" within the chapter "{topic}".

            CONTEXT FROM WEB RESEARCH:
            {knowledge_base if knowledge_base else "No web context available. Rely on your internal knowledge."}
            ---
            
            Generate:
            1. A list of essential prerequisite topics.
            2. A list of 3-5 key problem-solving tips.
            3. A list of 2-3 common student pitfalls.

            Return a single JSON object with keys: "prerequisites", "problem_solving_tips", "common_pitfalls".
            """
            topic_details_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": topic_details_prompt}],
                model="openai/gpt-oss-120b",
                response_format={"type": "json_object"},
            )
            topic_details = json.loads(
                topic_details_completion.choices[0].message.content
            )
            topic_obj.update(topic_details)
            print(f"    -> Details for topic '{topic_name}' populated.")

            # 4b: Populate reading material and MCQs for each concept
            for concept_obj in topic_obj["concepts"]:
                concept_name = concept_obj["concept_name"]
                print(f"    -> Generating content for Concept: '{concept_name}'")

                concept_content_prompt = f"""
                Act as a master teacher for IIT-JEE. Use the context below to generate content for the concept "{concept_name}" under the topic "{topic_name}".

                CONTEXT FROM WEB RESEARCH:
                {knowledge_base if knowledge_base else "No web context available. Rely on your internal knowledge."}
                ---

                Generate the following:
                1. "reading_material": A concise, 5-minute explanation of the concept, including formulas and definitions.
                2. "mcqs": An array of 3-4 high-quality MCQs testing this concept.

                Each MCQ must be a JSON object with: "question", "options" (array), "correct_answer_index" (integer), and "explanation".

                Return a single JSON object with the two keys: "reading_material" and "mcqs".
                """
                concept_content_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": concept_content_prompt}],
                    model="openai/gpt-oss-120b",
                    response_format={"type": "json_object"},
                )
                concept_content = json.loads(
                    concept_content_completion.choices[0].message.content
                )
                concept_obj.update(concept_content)

        # --- STEP 5: Synthesis Complete ---
        print("\n\n--- ✅ Learning Path Synthesis Complete! ---")
        return learning_path

    except Exception as e:
        print(f"\n--- ❌ An error occurred during learning path generation ---")
        print(f"Error: {repr(e)}")
        return None
