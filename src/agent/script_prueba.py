import os

# Importamos las funciones de los nodos que has creado
# (Asegúrate de que las rutas de importación coinciden con tus carpetas reales)
from nodes.node_llm import nodo_director_arte
from nodes.node_sam import nodo_segmentador


def ejecutar_test():
    print("🚀 INICIANDO TEST DE INTEGRACIÓN DE NODOS 🚀")
    print("=" * 50)
    
    # 1. PREPARAMOS EL ESTADO INICIAL (Simulando lo que enviaría el usuario)
    # IMPORTANTE: Cambia esta ruta por la de una imagen real de una zapatilla que tengas en tu Mac
    IMAGEN_PRUEBA = "src/cambio_fondo/img.png" 
    PROMPT_PRUEBA = "Pon esta botella en una calle cyberpunk lloviendo de noche"
    
    if not os.path.exists(IMAGEN_PRUEBA):
        print(f"❌ ERROR PREVIO: No se encuentra la imagen en {IMAGEN_PRUEBA}.")
        print("Por favor, pon una imagen real en esa ruta antes de ejecutar el test.")
        return

    # Este es nuestro "Tablero de Juego" virtual
    estado_actual = {
        "ruta_imagen_original": IMAGEN_PRUEBA,
        "prompt_usuario": PROMPT_PRUEBA
    }

    print(f"📦 ESTADO INICIAL:\n{estado_actual}\n")
    print("-" * 50)

    # ==========================================
    # TEST NODO 1: LLM (Director de Arte)
    # ==========================================
    print("⏳ Ejecutando Nodo 1 (LLM)...")
    
    # Ejecutamos el nodo pasándole el estado
    actualizacion_llm = nodo_director_arte(estado_actual)
    
    # Simulamos lo que hace LangGraph: fusionar el diccionario nuevo con el viejo
    estado_actual.update(actualizacion_llm)
    
    print("\n📦 ESTADO TRAS EL NODO 1:")
    print(f"- Prompt Optimizado: {estado_actual.get('prompt_optimizado')}")
    print(f"- Prompt Negativo: {estado_actual.get('prompt_negativo')}")
    print("-" * 50)

    # ==========================================
    # TEST NODO 2: SAM (Segmentador)
    # ==========================================
    print("\n⏳ Ejecutando Nodo 2 (SAM)...")
    
    # Ejecutamos el nodo SAM pasándole el estado (que ahora ya tiene los prompts, aunque SAM no los use)
    actualizacion_sam = nodo_segmentador(estado_actual)
    
    # Fusionamos de nuevo
    estado_actual.update(actualizacion_sam)
    
    print("\n📦 ESTADO TRAS EL NODO 2:")
    ruta_mascara = estado_actual.get('ruta_mascara_sam')
    print(f"- Ruta de la máscara: {ruta_mascara}")
    
    if ruta_mascara and os.path.exists(ruta_mascara):
        print("✅ ¡ÉXITO! La máscara se ha generado físicamente en el disco duro.")
    else:
        print("❌ FALLO: La máscara no se ha creado.")
        
    print("=" * 50)
    print("🎯 TEST FINALIZADO. SI VES LA MÁSCARA Y LOS PROMPTS, TODO FUNCIONA.")

if __name__ == "__main__":
    ejecutar_test()