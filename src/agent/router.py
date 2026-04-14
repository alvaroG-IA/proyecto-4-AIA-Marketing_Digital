import socket

from state import ProductEnvironmentState


def comprobar_internet() -> bool:
    """Hace un ping ultrarrápido a los servidores DNS de Cloudflare para ver si hay WiFi."""
    try:
        # Intenta conectar al puerto 53 de 1.1.1.1 con un tiempo límite de 2 segundos
        socket.create_connection(("1.1.1.1", 53), timeout=2)
        return True
    except OSError:
        return False


def enrutador_de_renderizado(state: ProductEnvironmentState) -> str:
    """
    Esta función NO modifica el estado. Solo mira el mundo exterior (o el estado)
    y devuelve un 'String' que LangGraph usará como mapa.
    """
    print("🚦 [Enrutador] Comprobando estado de la red...")

    # También podrías mirar una variable del estado si el usuario quiere forzar el modo local:
    # if state.get("forzar_modo_local") == True: return "ir_a_local"

    if comprobar_internet():
        print("🚦 [Enrutador] ✅ Internet detectado. Desviando tráfico a la Nube (replicate).")
        return "ir_a_local"
    else:
        print("🚦 [Enrutador] ⚠️ Sin conexión a Internet. Desviando tráfico a GPU Local (Diffusers).")
        return "ir_a_local"
