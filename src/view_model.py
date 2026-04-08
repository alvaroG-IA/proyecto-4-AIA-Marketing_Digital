import torch
import smplx
import numpy as np
import trimesh
import pyrender
from body_parameters import height_weight_to_betas

import pyrender
import trimesh

# Ruta al modelo SMPL-X
MODEL_PATH = "../models"
device = torch.device("cpu")

# Cargamos SMPL-X
smpl_model = smplx.create(
    model_path=MODEL_PATH,
    model_type="smplx",
    gender="male",
    num_betas=10,
    use_pca=False,
    ext="pkl"
).to(device)

def generate_smplx_mesh(height_cm, weight_kg):
    betas = height_weight_to_betas(height_cm, weight_kg)
    betas = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)

    body_pose = torch.zeros((1, smpl_model.NUM_BODY_JOINTS*3), dtype=torch.float32)
    global_orient = torch.zeros((1, 3), dtype=torch.float32)
    left_hand_pose = torch.zeros((1, 15*3), dtype=torch.float32)
    right_hand_pose = torch.zeros((1, 15*3), dtype=torch.float32)
    jaw_pose = torch.zeros((1, 3), dtype=torch.float32)
    expression = torch.zeros((1, 10), dtype=torch.float32)
    eye_pose = torch.zeros((1, 6), dtype=torch.float32)

    output = smpl_model(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        jaw_pose=jaw_pose,
        expression=expression,
        eye_pose=eye_pose,
        return_verts=True
    )

    vertices = output.vertices[0].detach().cpu().numpy()
    faces = smpl_model.faces
    return vertices, faces

def show_mesh_interactive(vertices, faces):
    # Crear el mesh de pyrender desde trimesh
    mesh = trimesh.Trimesh(vertices, faces)
    mesh = pyrender.Mesh.from_trimesh(mesh)

    # Crear una escena y añadir el mesh
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255])
    scene.add(mesh)

    # Abrir ventana interactiva
    pyrender.Viewer(scene, use_raymond_lighting=True)

if __name__ == "__main__":
    height = float(input("Altura (cm): "))
    weight = float(input("Peso (kg): "))

    verts, faces = generate_smplx_mesh(height, weight)
    show_mesh_interactive(verts, faces)