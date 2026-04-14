from typing import TypedDict, Optional


class ProductEnvironmentState(TypedDict):
    original_img_path: str              # Este campo guarda la ruta de la imágen inicial dada por el usuario
    user_prompt: str                    # Este campo guarda el prompt dado por el usuario con el entorno desado

    positive_prompt: Optional[str]      # Este campo guarda el promtp positivo generado tras la optimización del prompt original
    negative_prompt: Optional[str]      # Este campo guarda el promtp negativo generado tras la optimización del prompt original

    ruta_mascara_sam: Optional[str]     # Este campo guarda la ruta de la máscara obtenida por SAM del objeto
    ruta_resultado: Optional[str]       # Este campo guarda la ruta de la imagen final
