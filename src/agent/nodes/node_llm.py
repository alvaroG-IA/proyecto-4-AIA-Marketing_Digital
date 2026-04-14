# src/agent/pruebas/prompt_optimizer.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Dict, Any


class OptimizedPrompt(BaseModel):
    positive_prompt: str = Field(
        description="MAX 50 WORDS. Comma-separated keywords for Stable Diffusion."
    )
    negative_prompt: str = Field(
        description="Negative keywords to avoid in Stable Diffusion."
    )
    flux_prompt: str = Field(
        description="A highly descriptive, natural language paragraph (approx 80-100 words) explaining the exact same scene for the FLUX model."
    )


SYSTEM_PROMPT = """You are a Master Art Director for AI Photography.
Your job is to translate the user's idea into TWO different prompt styles simultaneously: 
one for Stable Diffusion (tags) and one for FLUX (natural language).

STRICT RULES:
1. EVERYTHING MUST BE IN ENGLISH. Translate the input if necessary.
2. 'positive_prompt': MAX 50 WORDS. Use ONLY comma-separated keywords (e.g., 50mm lens, 8k, cinematic lighting).
3. 'negative_prompt': Comma-separated keywords of things to avoid.
4. 'flux_prompt': A highly descriptive, flowing paragraph in natural language. Describe lighting, textures, mood, and camera angles explicitly. NO comma-separated lists here.

EXAMPLE:
Input: "zapatilla en la luna"
Output: {{ 
  "positive_prompt": "sneaker on lunar surface, earth in background, cinematic lighting, 50mm lens, photorealistic, 8k", 
  "negative_prompt": "ugly, blurry, atmosphere, clouds, deformed",
  "flux_prompt": "A highly detailed, photorealistic close-up of a modern sneaker resting on the dusty, cratered surface of the moon. In the background, a breathtaking view of the Earth is visible against the pitch-black starry sky. The scene is illuminated by dramatic, harsh directional sunlight typical of lunar photography, casting deep, sharp shadows."
}}
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{user_idea}")
])

llm = ChatOllama(model="llama3.2", temperature=0.1, num_predict=300)
extractor = PROMPT_TEMPLATE | llm.with_structured_output(OptimizedPrompt)


def nodo_director_arte(state: Dict[str, Any]) -> Dict[str, Any]:
    print("\n🧠 Optimizing prompt (Dual Mode: SD + FLUX)...")

    basic_idea = state.get("user_prompt") or state.get("prompt")

    if not basic_idea:
        return {}

    try:
        result = extractor.invoke({"user_idea": basic_idea})

        return {
            "positive_prompt": result.positive_prompt,
            "negative_prompt": result.negative_prompt,
            "flux_prompt": result.flux_prompt,
            "optimized_prompt": result.positive_prompt
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "positive_prompt": f"{basic_idea}, cinematic lighting, high quality, 8k",
            "negative_prompt": "blurry, distorted, low quality",
            "flux_prompt": f"A highly detailed, photorealistic image of {basic_idea}. The scene features beautiful cinematic lighting, sharp focus, and is shot in 8k resolution.",
            "optimized_prompt": f"{basic_idea}, cinematic lighting, high quality, 8k"
        }