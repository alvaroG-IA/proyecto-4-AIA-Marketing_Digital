import os
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

class PromptOptimizado(BaseModel):
    positive_prompt: str = Field(
        description="MANDATORY: Write ONLY IN ENGLISH. Use comma-separated keywords, NO narrative sentences. Include: lighting, camera, environment, resolution."
    )
    negative_prompt: str = Field(
        description="MANDATORY: Write ONLY IN ENGLISH. Comma-separated keywords of things to avoid."
    )

SYSTEM_PROMPT = """You are an expert Art Director for commercial product photography AI generation.
Your task is to translate the user's basic idea into a highly technical, comma-separated ENGLISH prompt for Stable Diffusion / FLUX.

RULES:
1. YOU MUST WRITE EVERYTHING IN ENGLISH. NO SPANISH.
2. DO NOT write sentences like "Imagine a street...". Use raw keywords separated by commas.
3. Always include camera terms (e.g., 50mm lens, DSLR, macro).
4. Always include lighting terms (e.g., cinematic lighting, studio lighting).
5. Always include quality terms (e.g., 8k, masterpiece, hyperrealistic).

EXAMPLE INPUT: "pon la zapatilla en una playa al atardecer"
EXAMPLE OUTPUT POSITIVE: "sneaker on white sand beach, golden hour sunset, warm cinematic lighting, ocean waves in background, 50mm lens, DSLR photography, photorealistic, 8k resolution, highly detailed"
EXAMPLE OUTPUT NEGATIVE: "ugly, artificial, illustration, text, watermark, bad anatomy, deformed"
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Idea del usuario (translate and optimize to english keywords): {idea_usuario}")
])

llm = ChatOllama(
    model="llama3.2", 
    temperature=0.4, 
    format="json"
)

extractor_agente = PROMPT_TEMPLATE | llm.with_structured_output(PromptOptimizado)


def nodo_director_arte(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo de LangGraph que coge la idea del usuario y genera los prompts técnicos.
    """
    print("\n[Nodo 1] 🧠 Director de Arte (Llama 3.2) analizando la idea...")
    
    # 1. Extraemos la idea original del estado global del grafo
    idea_basica = state.get("prompt_usuario", None)
    
    if not idea_basica:
        print("[Nodo 1] ⚠️ Advertencia: No se encontró un prompt de usuario en el estado.")
        return {}

    # 2. Ejecutamos la cadena (invocación al modelo local)
    resultado = extractor_agente.invoke({"idea_usuario": idea_basica})
    
    print("[Nodo 1] ✅ Prompts generados correctamente.")
    
    # 3. Devolvemos SOLO las variables del estado que hemos creado/modificado
    # LangGraph se encargará de inyectar esto de vuelta en el estado principal.
    return {
        "prompt_optimizado": resultado.positive_prompt,
        "prompt_negativo": resultado.negative_prompt
    }
