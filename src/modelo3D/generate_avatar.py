import torch
import smplx
from src.modelo3D.body_parameters import height_weight_to_betas

# Ruta donde está el archivo SMPL_MALE.pkl
MODEL_PATH = "../../models"

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

    body_pose = torch.zeros((1, 21, 3), dtype=torch.float32)

    # 🔹 Joints correctos SMPL-X
    NECK = 12
    HEAD = 14
    RIGHT_SHOULDER = 15
    LEFT_SHOULDER = 16
    RIGHT_ELBOW = 17
    LEFT_ELBOW = 18
    RIGHT_HAND = 19
    LEFT_HAND = 20

    # -------------------------------------------------
    # 1) Cabeza recta
    body_pose[0, NECK] = torch.tensor([0.0, 0.0, 0.0])
    body_pose[0, HEAD] = torch.tensor([0.0, 0.0, 0.0])

    # -------------------------------------------------
    # 2) Brazos relajados naturales (A-pose)
    arm_angle = 1.1

    # Separación lateral (eje Z)
    body_pose[0, LEFT_SHOULDER, 2] = arm_angle
    body_pose[0, RIGHT_SHOULDER, 2] = -arm_angle

    # Ligera rotación hacia delante (eje X)
    body_pose[0, LEFT_SHOULDER, 1] = 0.1
    body_pose[0, RIGHT_SHOULDER, 1] = -0.1

    # -------------------------------------------------
    # 3) Codos ligeramente doblados

    body_pose[0, LEFT_ELBOW, 1] = 0.25
    body_pose[0, RIGHT_ELBOW, 1] = -0.25

    body_pose[0, LEFT_ELBOW, 2] = 0.5
    body_pose[0, RIGHT_ELBOW, 2] = -0.5

    # 4) Manos bien puestas
    body_pose[0, LEFT_HAND, 0] = 0.2
    body_pose[0, RIGHT_HAND, 0] = -0.2

    body_pose[0, LEFT_HAND, 2] = -0.1
    body_pose[0, RIGHT_HAND, 2] = -0.1

    # -------------------------------------------------
    # Convertir a (1,63)
    body_pose = body_pose.view(1, -1)

    # -------------------------------------------------

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
    joints = output.joints[0].detach().cpu().numpy()
    faces = smpl_model.faces

    return vertices, faces, joints