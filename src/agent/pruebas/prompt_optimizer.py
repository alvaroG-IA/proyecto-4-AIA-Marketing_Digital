# tools/prompt_tool.py
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0.3)

def optimize_prompt_tool(state):
    print("🧠 Optimizing prompt...")

    prompt = state["prompt"]

    system = """
    Convert to highly detailed Stable Diffusion prompt in ENGLISH.
    Use: lighting, camera, realism, environment.
    Return ONLY keywords.
    """

    result = llm.invoke(system + "\nUser: " + prompt)

    return {"optimized_prompt": result.content}