# agent.py
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0.2)

TOOLS = {
    "optimize_prompt": "tool_1",
    "segment": "tool_2",
    "generate": "tool_3",
}


def agent_decision(state):
    # 1. Evaluamos el estado matemáticamente
    has_prompt = bool(state.get('optimized_prompt'))
    has_mask = bool(state.get('mask_path'))
    has_result = bool(state.get('result_path'))

    # 2. Construimos las etiquetas visuales para el LLM
    s1 = "[DONE]" if has_prompt else "[PENDING]"
    s2 = "[DONE]" if has_mask else "[PENDING]"
    s3 = "[DONE]" if has_result else "[PENDING]"
    s4 = "[PENDING]"

    # 3. Prompt de Lista de Tareas (Checklist)
    prompt = f"""You are a strict pipeline controller. Your job is to read the checklist below and output the exact name of the FIRST action marked as [PENDING].

CRITICAL RULE: YOU MUST NEVER REPEAT AN ACTION MARKED AS [DONE].

EXPECTED PIPELINE ORDER & CURRENT STATUS:
1. optimize_prompt : {s1}
2. segment         : {s2}
3. generate        : {s3}
4. finish          : {s4}

INSTRUCTION: 
Read the checklist from top to bottom. Find the FIRST action that says [PENDING]. 
Output ONLY that action name. No punctuation, no quotes, no explanations.

ACTION:"""

    print("\n--- PROMPT CHECKLIST ENVIADO AL ENRUTADOR ---")
    print(prompt)
    print("---------------------------------------------")

    # 4. Invocamos al LLM
    raw_action = llm.invoke(prompt).content

    # 5. Limpieza defensiva absoluta (Llama 3.2 es muy sucio escribiendo)
    action = raw_action.strip().replace(".", "").replace("`", "").replace('"', "").replace("'", "").lower()

    # Por si el modelo alucina y suelta una frase entera, cogemos solo la última palabra
    if " " in action:
        action = action.split()[-1]

    print(f"🤖 El LLM ha decidido ir al nodo: -> [{action}]")

    return action