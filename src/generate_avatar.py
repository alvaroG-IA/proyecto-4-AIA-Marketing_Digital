import torch
import smplx
import numpy as np
from body_parameters import height_weight_to_betas

# Ruta donde está el archivo SMPL_MALE.pkl
MODEL_PATH = "../models"

device = torch.device("cpu")

# usamos gender="male" porque tenemos SMPLX_MALE.pkl
smpl_model = smplx.create(
    model_path=MODEL_PATH,
    model_type="smplx",
    gender="male",          # <- antes estaba "neutral"
    num_betas=10,
    use_pca=False,
    ext="pkl"
).to(device)


def generate_smpl_mesh(height_cm, weight_kg):
    betas = height_weight_to_betas(height_cm, weight_kg)
    betas = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)

    # SMPL-X
    body_pose = torch.zeros((1, smpl_model.NUM_BODY_JOINTS * 3), dtype=torch.float32)  # 21*3 = 63
    global_orient = torch.zeros((1, 3), dtype=torch.float32)
    left_hand_pose = torch.zeros((1, 15*3), dtype=torch.float32)  # 15 joints mano
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