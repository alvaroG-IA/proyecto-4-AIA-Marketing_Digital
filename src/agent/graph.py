import socket
from langgraph.graph import StateGraph, START, END

from state import ProductEnvironmentState

from nodes.node_llm import nodo_director_arte
from nodes.node_sam import nodo_segmentador
from nodes.node_api import nodo_api_fal
from nodes.node_local import nodo_generador

from router import enrutador_de_renderizado


# ==========================================
# FUNCIÓN ENRUTADORA (El Guardia de Tráfico)
# ==========================================

def crear_agente_producto():
    print("🧠 Construyendo la arquitectura LangGraph con tolerancia a fallos...")

    workflow = StateGraph(ProductEnvironmentState)

    # 1. Añadimos todos los nodos posibles
    workflow.add_node("director_arte", nodo_director_arte)
    workflow.add_node("segmentador_sam", nodo_segmentador)
    workflow.add_node("pintor_nube", nodo_api_fal)
    workflow.add_node("pintor_local", nodo_generador)

    # 2. Las conexiones estáticas (Lo que siempre ocurre igual)
    workflow.add_edge(START, "director_arte")
    workflow.add_edge("director_arte", "segmentador_sam")

    # 3. LA BIFURCACIÓN (Conditional Edges)
    workflow.add_conditional_edges(
        "segmentador_sam",  # Nodo de salida
        enrutador_de_renderizado,  # Función que toma la decisión
        {
            "ir_a_nube": "pintor_nube",
            "ir_a_local": "pintor_local"
        }
    )

    # 4. Los finales (Ambos caminos terminan el programa)
    workflow.add_edge("pintor_nube", END)
    workflow.add_edge("pintor_local", END)

    return workflow.compile()
