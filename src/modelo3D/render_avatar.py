import pyrender
import trimesh
import numpy as np
import imageio


def create_scene(mesh):
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    scene.add(mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    # Pose de cámara centrada
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -0.4],
        [0, 0, 1, 2.5],
        [0, 0, 0, 1],
    ])
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    scene.add(light, pose=camera_pose)
    return scene


def render_full_and_masks(vertices, faces, parts_dict, output_folder="."):
    """
    Renderiza el avatar completo y todas las máscaras especificadas en parts_dict.
    """
    renderer = pyrender.OffscreenRenderer(512, 768)

    # 1. Render Avatar Principal
    main_mesh = trimesh.Trimesh(vertices, faces)
    pn_full_mesh = pyrender.Mesh.from_trimesh(main_mesh)
    scene_full = create_scene(pn_full_mesh)
    color, _ = renderer.render(scene_full)
    imageio.imwrite(f"{output_folder}/avatar.png", color)

    # Material para máscaras: Blanco puro y Doble Cara (Evita puntos negros)
    mask_material = pyrender.MetallicRoughnessMaterial(
        doubleSided=True,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0]
    )

    # 2. Renderizar Máscaras
    for part_name, v_indices in parts_dict.items():
        # Filtro robusto: Cara incluida si cualquiera de sus vértices está en la lista
        face_mask = np.isin(faces, v_indices).any(axis=1)
        part_faces = faces[face_mask]

        if len(part_faces) == 0:
            continue

        part_trimesh = trimesh.Trimesh(vertices, part_faces)
        pn_part_mesh = pyrender.Mesh.from_trimesh(part_trimesh, material=mask_material)

        scene_part = create_scene(pn_part_mesh)
        # Renderizamos el color (que será blanco)
        mask_color, _ = renderer.render(scene_part)

        # Convertir a binario (Gris -> Blanco)
        mask_final = (mask_color[..., 0] > 0).astype(np.uint8) * 255
        imageio.imwrite(f"{output_folder}/mask_{part_name}.png", mask_final)

    renderer.delete()