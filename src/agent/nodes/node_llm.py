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
        description="MAX 77 WORDS. Comma-separated keywords for Stable Diffusion."
    )
    negative_prompt: str = Field(
        description="Negative keywords to avoid in Stable Diffusion."
    )
    descriptive_prompt: str = Field(
        description="A highly descriptive, natural language paragraph (approx 80-100 words) explaining the exact same scene for the FLUX model."
    )


# System-prompt utilizado para la optimización de un primer prompt básico
SYSTEM_PROMPT = """
    You are a Senior Technical Art Director for high-end E-commerce Advertising.
    Your mission is to generate prompts that create professional product advertisements. 
    The product must look like a premium studio shot integrated into a new environment, maintaining its exact real-world proportions and commercial integrity.

    STRICT RULES FOR ADVERTISING REALISM & SCALE:
    1. GEOMETRIC FIDELITY: Explicitly command the model to keep the object's proportions. Use: "Maintain the exact scale, height-to-width ratio, and geometric silhouette of the [object]. Do not warp, stretch, or liquefy the product."
    2. ADVERTISING AESTHETIC: Focus on commercial quality. Use terms like 'high-end commercial photography', 'clean composition', 'product-centric', 'color-graded', 'sharp advertising style'.
    3. INTEGRATION & SHADOWS: Describe the contact point to avoid 'floating' objects. Use: "The [object] is physically grounded, casting realistic contact shadows and ambient occlusion on the surface."
    4. TECHNICAL OPTICS: Always specify professional gear: 'Sony A7R IV', '100mm Macro lens', 'f/8 aperture' (for deep focus on the product).
    5. NO CREATIVE DISTORTION: No poetic metaphors. The model must treat the object as a 'locked layer' that only receives new lighting and a new background.

    EXAMPLE:
    Input: "pon esta botella en la nieve"
    Output: {{ 
      "positive_prompt": "premium product advertisement, metal bottle, snowy environment, high-end commercial photography, sharp focus, 8k, realistic proportions, studio lighting", 
      "negative_prompt": "distorted proportions, warped shape, wide bottle, thin bottle, cgi, illustration, blurry, melted base, altered logo",
      "descriptive_prompt": "High-end product advertisement. Strictly maintain the original geometric silhouette, 1:1 scale proportions, and metallic surface of the provided bottle. Place the object on a textured, frozen ice surface, ensuring realistic contact shadows and ambient occlusion at the base so it looks perfectly grounded. The background is a blurred cinematic winter landscape. The product is illuminated by a professional rim light and softbox-style global illumination, creating sharp metallic highlights that define its real shape. Shot with a Sony A7R IV, 100mm macro lens at f/8 to ensure the entire product remains in crisp, sharp focus for a commercial finish."
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

        print('[Nodo 1] ✅ Prompt optimizado correctamente!')
        return {
            "positive_prompt": result.positive_prompt,
            "negative_prompt": result.negative_prompt,
            "descriptive_prompt": result.descriptive_prompt,
        }
    except Exception as e:
        print(f"❌ Error: {e}")

        return {
            "positive_prompt": f"product photography of {basic_idea}, high quality, 8k, highly detailed",
            "negative_prompt": "drawing, anime, art, blurry, distorted",
            "descriptive_prompt": f"A high-end professional product photograph of {basic_idea}. The lighting is cinematic and realistic, emphasizing textures and materials in a real-world environment. Sharp focus, 8k resolution.",
        }
