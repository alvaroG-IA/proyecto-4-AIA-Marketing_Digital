from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

SYSTEM_PROMPT = """
You are a senior Stable Diffusion prompt engineer.

You MUST generate prompts in STRICT TAG FORMAT.

RULES:
- NO sentences.
- NO poetry.
- NO full stops.
- ONLY comma-separated keywords.
- ONLY English.
- DO NOT use line breaks.
- DO NOT describe like a story.
- Replace vague words like "product" with "product photography" or "studio product photography
- Avoid single-word generic tokens unless necessary
- Prefer descriptive photography phrases

STRUCTURE:
1. scene
2. environment
3. lighting
4. camera
5. photography settings
6. quality tags

ALWAYS include:
- professional product photography
- photorealistic
- ultra detailed
- 8k

Return ONLY the final prompt.
"""

def optimize_prompt(user_prompt_es):

    chat = f"""<s>[INST] {SYSTEM_PROMPT}

User request in Spanish:
{user_prompt_es}

English prompt:
[/INST]"""

    inputs = tokenizer(chat, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 🔥 Limpieza robusta del output
    final_prompt = decoded.split("English prompt:")[-1].strip()
    final_prompt = re.split(r"\[/INST\]", final_prompt)[-1].strip()

    return final_prompt