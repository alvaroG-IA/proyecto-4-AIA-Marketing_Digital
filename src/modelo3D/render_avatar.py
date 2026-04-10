import pyrender
import trimesh
import numpy as np

def render_mesh(vertices, faces, output_path="avatar.png"):
    mesh = trimesh.Trimesh(vertices, faces)

    mesh = pyrender.Mesh.from_trimesh(mesh)

    scene = pyrender.Scene(bg_color=[255,255,255,255])
    scene.add(mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -0.2],
        [0, 0, 1, 2.4],
        [0, 0, 0, 1],
    ])
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(512, 768)
    color, _ = renderer.render(scene)

    import imageio
    imageio.imwrite(output_path, color)