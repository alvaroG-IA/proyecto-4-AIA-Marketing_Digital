import torch
import numpy as np
from render_avatar import render_full_and_masks
from generate_avatar import generate_smpl_mesh, smpl_model


def get_segmentation_by_joints(vertices):
    """
    Usa los pesos de skinning del modelo para segmentar.
    SMPL-X tiene 55 joints.
    """
    # Obtenemos los pesos (lbl_weights) que definen qué joint influye en qué vértice
    # lbs_weights shape: (10475, 55)
    weights = smpl_model.lbs_weights.detach().cpu().numpy()

    # Para cada vértice, vemos qué articulación tiene más peso
    vertex_joint_assignment = np.argmax(weights, axis=1)

    # Agrupamos según los IDs oficiales de SMPL-X
    # 0:Pelvis, 1:L_Hip, 2:R_Hip, 3:Spine1, 4:L_Knee, 5:R_Knee, 6:Spine2...
    # 12:Neck, 15:R_Shoulder, 16:L_Shoulder, 17:R_Elbow, 18:L_Elbow, 20:L_Wrist, 21:R_Wrist

    parts = {
        # CABEZA: Cuello (12), Mandíbula (15), Globos oculares
        "head": np.where(np.isin(vertex_joint_assignment, [12, 15, 22, 23, 24, 52, 53, 54]))[0],

        # TORSO Y BRAZOS: Spine (3,6,9), Hombros (16,17), Codos (18,19), Muñecas (20,21)
        "torso_arms": np.where(np.isin(vertex_joint_assignment, [
            0, 3, 6, 9,  # Tronco central / Pelvis
            13, 14, 16, 17,  # Clavículas y Hombros
            18, 19, 20, 21,  # Brazos, Codos y Muñecas
            25, 26, 27, 28, 29,  # Dedos mano izq
            40, 41, 42, 43, 44  # Dedos mano der
        ]))[0],

        # PIERNAS: Caderas (1,2) y Rodillas (4,5)
        "legs": np.where(np.isin(vertex_joint_assignment, [1, 2, 4, 5]))[0],

        # PIES: Tobillos (7,8) y Dedos pie (10,11)
        "feet": np.where(np.isin(vertex_joint_assignment, [7, 8, 10, 11]))[0]
    }
    return parts


def main():
    h = float(input("Altura (cm): "))
    w = float(input("Peso (kg): "))

    # 1. Generar
    vertices, faces, joints = generate_smpl_mesh(h, w)

    # 2. Segmentar por pesos de articulación
    print("Segmentando usando pesos de skinning (LBS)...")
    parts_dict = get_segmentation_by_joints(vertices)

    # 3. Renderizar
    # (Asegúrate de que render_full_and_masks use el método robusto que vimos antes)
    render_full_and_masks(vertices, faces, parts_dict)

    print("✔ Máscaras generadas basadas en la estructura ósea del modelo.")


if __name__ == "__main__":
    main()