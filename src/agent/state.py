from typing import TypedDict, Optional

class ProductEnvironmentState(TypedDict):
    ruta_imagen_original: str           # Este campo guarda la ruta de la imágen inicial dada por el usuario 
    prompt_usuario: str                 # Este campo guarda el promt dado por el usuari con el entorno desado

    prompt_optimizado: Optional[str]    # Este campo guarda el promtp positivo generado tras la optimización del prompt original
    prompt_negativo: Optional[str]      # Este campo guarda el promtp negativo generado tras la optimización del prompt original

    ruta_mascara_sam: Optional[str]     # Este campo guarda la ruta de la máscara obtenida por SAM del objeto
    ruta_imagen_final: Optional[str]    # Este campo guarda la ruta de la imagen final
