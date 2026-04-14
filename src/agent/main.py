import os
from graph import crear_agente_producto


def main():
    print("=" * 60)
    print("🚀 INICIANDO AGENTE DE PRODUCTO E-COMMERCE 🚀")
    print("=" * 60)

    # 1. Selección de imagen y prompt del usuario
    USER_IMG = "data/images/botella.png"
    PROMPT_USUARIO = input('Que necesitas hacer?: ')

    if not os.path.exists(USER_IMG):
        print(f"❌ Error: No se encuentra la imagen en {USER_IMG}.")
        return

    # 2. Creación del estado inicial
    estado_inicial = {
        "original_img_path": USER_IMG,
        "user_prompt": PROMPT_USUARIO
    }

    # 3. Compilación del grafo
    agente = crear_agente_producto()

    print("\n⚡ Ejecutando pipeline (Director de Arte -> SAM -> Render)...")
    print("-" * 60)

    try:
        estado_final = agente.invoke(estado_inicial)

        # 4. Resumen final de la ejecución
        print("-" * 60)
        print("✅ PROCESO COMPLETADO CON ÉXITO")
        print("\n📦 RESULTADOS DEL ESTADO FINAL:")
        print(f"🎨 Prompt LLM: {estado_final.get('positive_prompt')}")
        print(f"🎨 Prompt FLUX-LLM: {estado_final.get('flux_prompt')}")
        print(f"✂️ Máscara SAM: {estado_final.get('sam_mask_path')}")
        print(f"📸 Imagen Final: {estado_final.get('final_img_path')}")

    except Exception as e:
        print(f"\n❌ El Agente sufrió un error durante la ejecución: {e}")


if __name__ == "__main__":
    main()
