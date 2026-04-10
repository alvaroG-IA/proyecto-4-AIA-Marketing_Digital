from src.modelo3D.generate_avatar import generate_smpl_mesh
from render_avatar import render_mesh

def main():
    height = float(input("Altura (cm): "))
    weight = float(input("Peso (kg): "))

    vertices, faces = generate_smpl_mesh(height, weight)
    render_mesh(vertices, faces)

    print("Avatar generado: avatar.png")

if __name__ == "__main__":
    main()