# agent.py
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0.2)

TOOLS = {
    "optimize_prompt": "tool_1",
    "segment": "tool_2",
    "generate": "tool_3",
}

def agent_decision(state):
    """
    El LLM decide qué hacer a continuación
    """

    prompt = f"""
You are an AI agent controlling a pipeline.

STATE:
{state}

You can choose ONLY one action:
- optimize_prompt
- segment
- generate
- finish

Return ONLY the action name.
"""

    action = llm.invoke(prompt).content.strip()

    return action