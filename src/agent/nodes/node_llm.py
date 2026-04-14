# src/agent/pruebas/prompt_optimizer.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Dict, Any


class OptimizedPrompt(BaseModel):
    positive_prompt: str = Field(default="", description="Technical keywords in English.")
    negative_prompt: str = Field(default="ugly, blurry", description="Negative keywords.")


SYSTEM_PROMPT = """You are a technical Art Director. 
Convert the user idea into a JSON with keywords in ENGLISH.

STRICT RULES:
1. MAX 50 WORDS for the positive_prompt. 
2. Use ONLY keywords separated by commas. NO sentences.
3. Include: cinematic lighting, 50mm lens, 8k, photorealistic.

EXAMPLE:
Input: "zapatilla en la luna"
Output: {{ "positive_prompt": "sneaker on lunar surface, earth in background, cinematic lighting", "negative_prompt": "blurry, atmosphere" }}
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{user_idea}")
])

llm = ChatOllama(model="llama3.2", temperature=0.1, num_predict=150)
extractor = PROMPT_TEMPLATE | llm.with_structured_output(OptimizedPrompt)


def nodo_director_arte(state: Dict[str, Any]) -> Dict[str, Any]:
    print("\n🧠 Optimizing prompt (Fixed Brackets)...")
    basic_idea = state.get("user_prompt")
    if not basic_idea:
        return {}

    try:
        result = extractor.invoke({"user_idea": basic_idea})
        return {
            "positive_prompt": result.positive_prompt,
            "negative_prompt": result.negative_prompt
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "positive_prompt": f"{basic_idea}, cinematic lighting, high quality, 8k",
            "negative_prompt": "blurry, distorted, low quality"
        }
