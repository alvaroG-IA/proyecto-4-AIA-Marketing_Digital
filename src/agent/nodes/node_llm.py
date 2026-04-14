# src/agent/pruebas/prompt_optimizer.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Dict, Any

# ==========================================
# 1. CONFIGURACIÓN Y CARGA GLOBAL DEL MODELO
# ==========================================
class OptimizedPrompt(BaseModel):
    """
    Clase encargada de establecer el tipo de salida deseada para el LLM optimizador de prompts
    """
    positive_prompt: str = Field(
        description="MAX 50 WORDS. Comma-separated keywords for Stable Diffusion."
    )
    negative_prompt: str = Field(
        description="Negative keywords to avoid in Stable Diffusion."
    )
    flux_prompt: str = Field(
        description="A highly descriptive, natural language paragraph (approx 80-100 words) explaining the exact same scene for the FLUX model."
    )


# System-prompt utilizado para la optimización de un primer prompt básico
SYSTEM_PROMPT = """
    You are a Technical Art Director for high-end E-commerce Photography.
    Your goal is to generate professional, hyper-realistic descriptions. Avoid poetic, mystical, or metaphorical language (NO "sentinels", "ghostly", "whispering trees").
    
    STRICT RULES FOR REALISM:
    1. FOCUS ON LIGHTING: Use terms like 'softbox lighting', 'rim light', 'global illumination', 'depth of field', 'ray-tracing'.
    2. DESCRIBE MATERIALS: Mention 'brushed metal texture', 'reflections', 'water droplets', 'detailed sand grains'.
    3. PHOTOGRAPHIC STYLE: Always specify a camera (e.g., Sony A7R IV), a lens (e.g., 85mm f/1.8 macro), and sharp focus.
    4. FLUX_PROMPT: Write a clean, technical description of a real photo. Do NOT use fantasy or storytelling elements. Focus on how the object is integrated into the environment.
    
    EXAMPLE:
    Input: "botella en la playa"
    Output: {{ 
      "positive_prompt": "professional product photography, metal bottle, black sand beach, sharp focus, 8k, cinematic lighting, macro lens", 
      "negative_prompt": "cgi, illustration, cartoon, fake, blurry, low resolution",
      "flux_prompt": "A professional commercial photograph of a sleek metal bottle standing on wet black volcanic sand. The sun is low on the horizon, creating high-contrast golden hour lighting with sharp reflections on the bottle's metallic surface. Small crystal-clear ocean waves with realistic foam gently lap around the base. Shot with a 100mm macro lens, f/2.8, showing extreme detail in the sand grains and water droplets. Photorealistic, 8k, highly detailed textures."
    }}
"""

# Generación de plantilla de chat basada en el system-prompt y la idea básica del usuario
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{user_idea}")
])

# Defición del modelo LLM
llm = ChatOllama(model="llama3.2", temperature=0.2, num_predict=300)

# Diseño de flujo de generación de salida esperada (usa la salida del LLM y le aplica la lógica de la clase encargada de definir el tipo de salida
extractor = PROMPT_TEMPLATE | llm.with_structured_output(OptimizedPrompt)


# ==========================================
# 2. FUNCIÓN NODO
# ==========================================
def nodo_optimizador_prompt(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo que recibe una idea básica del usuario y utiliza un LLM (llama3.2) para optimizarla y generar nuevos prompts.
    """
    print("[NODE 1]🧠 Optimizando prompt inicial del usuario...")

    basic_idea = state.get("user_prompt")

    if not basic_idea:
        return {}
    try:
        result = extractor.invoke({"user_idea": basic_idea})

        return {
            "positive_prompt": result.positive_prompt,
            "negative_prompt": result.negative_prompt,
            "flux_prompt": result.flux_prompt,
        }
    except Exception as e:
        print(f"❌ Error: {e}")

        return {
            "positive_prompt": f"product photography of {basic_idea}, high quality, 8k, highly detailed",
            "negative_prompt": "drawing, anime, art, blurry, distorted",
            "flux_prompt": f"A high-end professional product photograph of {basic_idea}. The lighting is cinematic and realistic, emphasizing textures and materials in a real-world environment. Sharp focus, 8k resolution.",
        }
