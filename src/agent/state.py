from typing import TypedDict, Optional


class ProductEnvironmentState(TypedDict):
    original_img_path: str              # Este campo guarda la ruta de la imágen inicial dada por el usuario
    user_prompt: str                    # Este campo guarda el prompt dado por el usuario con el entorno desado

    positive_prompt: Optional[str]      # Este campo guarda el promtp positivo generado tras la optimización del prompt original
    negative_prompt: Optional[str]      # Este campo guarda el promtp negativo generado tras la optimización del prompt original
    flux_prompt: Optional[str]          # Este campo guarda el prompt detallado que se usará en el caso de seleccionar el modelo de generación potente mediatne API

    sam_mask_path: Optional[str]        # Este campo guarda la ruta de la máscara obtenida por SAM del objeto
    final_img_path: Optional[str]       # Este campo guarda la ruta de la imagen final
