import numpy as np

def height_weight_to_betas(height_cm, weight_kg):
    """
    Convierte altura y peso en parámetros de forma SMPL (betas).
    Aproximación heurística basada en BMI.
    """

    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    betas = np.zeros(10)

    # beta[0] controla delgadez ↔ corpulencia
    betas[0] = (bmi - 22) / 6.0

    # beta[1] ancho de hombros
    betas[1] = betas[0] * 0.5

    # beta[2] volumen del torso
    betas[2] = betas[0] * 0.8

    # beta[3] volumen piernas
    betas[3] = betas[0] * 0.6

    # los demás pequeños ajustes aleatorios suaves
    betas[4:] = np.random.normal(0, 0.03, 6)

    return betas