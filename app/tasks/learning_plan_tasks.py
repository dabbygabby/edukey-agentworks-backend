import os
from groq import Groq
from pydantic import BaseModel, Field
from app.core.celery_app import celery_app
import json

# Pydantic model for the task payload
class LearningPlanTaskPayload(BaseModel):
    topic: str = Field(..., example="Explain the importance of low-latency LLMs")

@celery_app.task
def create_learning_plan(topic: str):
    """
    Orchestrates a multi-step process with the Groq API to generate a learning plan.

    Args:
        topic: The topic for which to create the learning plan.

    Returns:
        A dictionary containing the generated course plan, lesson plan, and sources,
        or None if an error occurs.
    """
    try:
        # It's recommended to set the GROQ_API_KEY in your environment variables.
        # You can also pass it directly: Groq(api_key="YOUR_API_KEY")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
            
        client = Groq(api_key=api_key)
        print(f"Starting learning plan generation for topic: \"{topic}\"")

        # --- STEP 1: Search for relevant URLs using a powerful model ---
        print("\n--- STEP 1: Searching for relevant sources... ---")
        
        search_prompt = (
            f'Find 3-5 relevant URLs for learning about "{topic}" for Indian competitive exams. '
            'Output only a JSON object with a single key "urls" containing an array of strings, '
            'like {"urls": ["url1", "url2"]}.'
        )
        
        search_completion = client.chat.completions.create(
            model="openai/gpt-oss-120b", # Using a powerful model for search and synthesis
            messages=[{"role": "user", "content": search_prompt}],
            response_format={"type": "json_object"},
        )
        
        search_result_content = search_completion.choices[0].message.content
        if not search_result_content:
            print("Error: Search step returned no content.")
            return None
            
        urls_data = json.loads(search_result_content)
        urls = urls_data.get("urls", [])
        
        if not urls:
            print("Could not find any relevant URLs. Exiting.")
            return None
            
        print(f"Found {len(urls)} sources: {urls}")

        # --- STEP 2: Visit each URL using the Compound tool to get content ---
        # Note: The 'groq/compound' model is a conceptual tool in the JS example.
        # In practice, you might need a different tool or model that can access URLs.
        # For this example, we'll simulate asking a model to summarize based on the URL.
        # Real-world URL fetching would require libraries like `requests` and `BeautifulSoup`.
        # Groq's tool-enabled models can handle this more directly if configured.
        
        print("\n--- STEP 2: Visiting and analyzing content from each source... ---")
        visited_content = []
        
        for url in urls:
            print(f"Analyzing content from: {url}")
            # This is a conceptual step. Groq API itself doesn't directly visit URLs
            # unless using a specific tool-enabled model. We ask it to act as if it did.
            visit_prompt = f"Please provide a concise summary of the likely content from this URL: {url}"
            
            # Using a faster model for summarization tasks
            visit_completion = client.chat.completions.create(
                model="openai/gpt-oss-120b", 
                messages=[{"role": "user", "content": visit_prompt}],
            )
            
            summary = visit_completion.choices[0].message.content or "Could not retrieve summary."
            visited_content.append({"url": url, "content": summary})
        
        print("Finished analyzing all sources.")

        # --- STEP 3: Synthesize the final lesson plan ---
        print("\n--- STEP 3: Synthesizing the final lesson plan... ---")
        
        synthesis_prompt = f"""
        Based on the following summarized content from various websites, create a learning plan for the topic "{topic}".
        The content is: {json.dumps(visited_content, indent=2)}
        
        Generate a pure JSON object with the following shape:
        {{
          "coursePlan": ["Week 1: Topic A...", "Week 2: Topic B...", ...],
          "lessonPlan": [
            {{ "title": "Lesson 1 Title", "description": "A brief description of the lesson." }},
            {{ "title": "Lesson 2 Title", "description": "Another lesson description." }}
          ],
          "sources": [
            {{ "title": "Source Page Title 1", "url": "url1" }},
            {{ "title": "Source Page Title 2", "url": "url2" }}
          ]
        }}
        - The lesson plan should have a maximum of 10 lessons.
        - The "sources" array MUST be populated using the original URLs. For the title, use a descriptive name based on the content or URL.
        """

        final_completion = client.chat.completions.create(
            model="openai/gpt-oss-120b", # Using a powerful model for the final synthesis
            messages=[{"role": "user", "content": synthesis_prompt}],
            response_format={"type": "json_object"},
        )
        
        final_result_content = final_completion.choices[0].message.content
        if not final_result_content:
            print("Error: Synthesis step returned no content.")
            return None

        final_plan = json.loads(final_result_content)
        print("\n--- ✅ Synthesis Complete! ---")
        
        return final_plan

    except Exception as e:
        print(f"\n--- ❌ An error occurred ---")
        print(f"Error: {e}")
        return None