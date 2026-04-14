from langgraph.graph import StateGraph, START, END

from state import ProductEnvironmentState

from nodes.node_llm import nodo_director_arte
from nodes.node_sam import nodo_segmentador
from nodes.node_api import nodo_api_replicate
from nodes.node_local import nodo_generador

from router import enrutador_de_renderizado


def crear_agente_producto():
    print("🧠 Construyendo la arquitectura LangGraph...")

    workflow = StateGraph(ProductEnvironmentState)

    # 1. Adición de todos los nodos disponibles
    workflow.add_node("director_arte", nodo_director_arte)
    workflow.add_node("segmentador_sam", nodo_segmentador)
    workflow.add_node("pintor_nube", nodo_api_replicate)
    workflow.add_node("pintor_local", nodo_generador)

    # 2. Definimos las conexiones estáticas
    workflow.add_edge(START, "director_arte")
    workflow.add_edge("director_arte", "segmentador_sam")

    # 3. Definimos la conexión condicional basada en la disposición de conexión a internet
    workflow.add_conditional_edges(
        "segmentador_sam",
        enrutador_de_renderizado,
        {
            "ir_a_nube": "pintor_nube",
            "ir_a_local": "pintor_local"
        }
    )

    # 4. Definición de los finales del flujo
    workflow.add_edge("pintor_nube", END)
    workflow.add_edge("pintor_local", END)

    return workflow.compile()
