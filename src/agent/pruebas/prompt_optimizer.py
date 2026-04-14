# tools/prompt_tool.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# 1. Definimos la estructura exacta que queremos que el LLM devuelva
class PromptOptimizado(BaseModel):
    positive_prompt: str = Field(
        description="MANDATORY: Write ONLY IN ENGLISH. Use comma-separated keywords, NO narrative sentences. Include: lighting, camera, environment, resolution."
    )
    negative_prompt: str = Field(
        description="MANDATORY: Write ONLY IN ENGLISH. Comma-separated keywords of things to avoid."
    )


# 2. El Prompt del Sistema con instrucciones estrictas
SYSTEM_PROMPT = """You are an expert Art Director for commercial product photography AI generation.
Your task is to translate the user's basic idea into a highly technical, comma-separated ENGLISH prompt.

RULES:
1. YOU MUST WRITE EVERYTHING IN ENGLISH. NO SPANISH.
2. DO NOT write narrative sentences. Use raw keywords separated by commas.
3. Include camera terms (e.g., 50mm lens, DSLR).
4. Include lighting terms (e.g., cinematic lighting, studio lighting).
5. Include quality terms (e.g., 8k, masterpiece, photorealistic).

EXAMPLE INPUT: "pon la zapatilla en una playa"
EXAMPLE OUTPUT POSITIVE: "sneaker on white sand beach, golden hour sunset, warm cinematic lighting, 50mm lens, photorealistic, 8k"
EXAMPLE OUTPUT NEGATIVE: "ugly, artificial, illustration, text, watermark, deformed"
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "User Idea: {idea_usuario}")
])

# Inicializamos el modelo forzando el formato JSON
llm = ChatOllama(
    model="llama3.2",
    temperature=0.2,
    format="json"
)

# Conectamos el prompt, el modelo y el parseador de Pydantic
extractor_agente = PROMPT_TEMPLATE | llm.with_structured_output(PromptOptimizado)


def optimize_prompt_tool(state):
    print("\n🧠 Optimizing prompt with Structured Output...")

    # Extraemos el prompt original del estado
    idea_basica = state.get("prompt")

    if not idea_basica:
        print("❌ Error: No prompt found in state.")
        return {}

    # Invocamos la cadena
    try:
        resultado = extractor_agente.invoke({"idea_usuario": idea_basica})
        print("✅ Prompts successfully extracted!")

        # Devolvemos ambas variables para que se guarden en el estado
        return {
            "optimized_prompt": resultado.positive_prompt,
            "negative_prompt": resultado.negative_prompt
        }

    except Exception as e:
        print(f"❌ Error during prompt optimization: {e}")
        return {}