import os
from graph import crear_agente_producto


def main():
    print("=" * 60)
    print("🚀 INICIANDO AGENTE DE PRODUCTO E-COMMERCE 🚀")
    print("=" * 60)

    IMAGEN_USUARIO = "src/cambio_fondo/img.png"
    PROMPT_USUARIO = input('Que necesitas hacer?: ')

    if not os.path.exists(IMAGEN_USUARIO):
        print(f"❌ Error: No se encuentra la imagen en {IMAGEN_USUARIO}.")
        return

    # 2. Creamos el estado inicial
    estado_inicial = {
        "original_img_path": IMAGEN_USUARIO,
        "user_prompt": PROMPT_USUARIO
    }

    # 3. Compilamos tu grafo
    agente = crear_agente_producto()

    print("\n⚡ Ejecutando pipeline (Director de Arte -> SAM -> Render)...")
    print("-" * 60)

    try:
        estado_final = agente.invoke(estado_inicial)

        print("-" * 60)
        print("✅ PROCESO COMPLETADO CON ÉXITO")
        print("\n📦 RESULTADOS DEL ESTADO FINAL:")
        print(f"🎨 Prompt LLM: {estado_final.get('prompt_optimizado')}")
        print(f"✂️ Máscara SAM: {estado_final.get('ruta_mascara_sam')}")
        print(f"📸 Imagen Final: {estado_final.get('ruta_resultado')}")

    except Exception as e:
        print(f"\n❌ El Agente sufrió un error durante la ejecución: {e}")


if __name__ == "__main__":
    main()