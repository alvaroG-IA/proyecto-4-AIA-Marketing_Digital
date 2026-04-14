# src/agent/pruebas/prompt_optimizer.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Dict, Any

class PromptOptimizado(BaseModel):
    positive_prompt: str = Field(default="", description="Technical keywords in English.")
    negative_prompt: str = Field(default="ugly, blurry", description="Negative keywords.")

# HE CORREGIDO LAS LLAVES AQUÍ USANDO {{ }}
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
    ("human", "{idea_usuario}")
])

llm = ChatOllama(model="llama3.2", temperature=0.1, num_predict=150)
extractor_agente = PROMPT_TEMPLATE | llm.with_structured_output(PromptOptimizado)


def optimize_prompt_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    print("\n🧠 Optimizing prompt (Fixed Brackets)...")
    idea_basica = state.get("prompt")
    if not idea_basica: return {}

    try:
        resultado = extractor_agente.invoke({"idea_usuario": idea_basica})
        return {
            "optimized_prompt": resultado.positive_prompt,
            "negative_prompt": resultado.negative_prompt
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "optimized_prompt": f"{idea_basica}, cinematic lighting, high quality, 8k",
            "negative_prompt": "blurry, distorted, low quality"
        }