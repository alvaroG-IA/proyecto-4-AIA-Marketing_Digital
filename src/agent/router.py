import socket

from state import ProductEnvironmentState


def comprobar_internet() -> bool:
    """Función que hace un ping rápido para comprobar si hay Wifi."""
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2)
        return True
    except OSError:
        return False


def enrutador_de_renderizado(state: ProductEnvironmentState) -> str:
    """
    Esta función comprueba si hay conexión a internet y devuelve un string que nuestro grafo LangGraph
    usará como mapa para decidir que nodo elegir.
    """
    print("\n[Enrutador] 🚦 Comprobando estado de la red...")

    if comprobar_internet():
        print("[Enrutador] ✅ Internet detectado. Desviando tráfico a la Nube (replicate).")
        return "ir_a_nube"
    else:
        print("[Enrutador] ⚠️ Sin conexión a Internet. Desviando tráfico a GPU Local (Diffusers).")
        return "ir_a_local"
