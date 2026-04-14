# graph.py
from prompt_optimizer import optimize_prompt_tool
from sam_segmentation import sam_tool
from diffuser_tool import diffusion_tool
from agente_orquestador import agent_decision

def run_agent(state):

    while True:

        action = agent_decision(state)

        print(f"🤖 Action: {action}")

        if action == "optimize_prompt":
            state.update(optimize_prompt_tool(state))

        elif action == "segment":
            state.update(sam_tool(state))

        elif action == "generate":
            state.update(diffusion_tool(state))

        elif action == "finish":
            break

    return state