# main.py
from graph import run_agent

if __name__ == "__main__":

    state = {
        "image_path": "src/cambio_fondo/img_3.png",
        "prompt": "Pon este objeto de manera aislada sobre una mesa blanca."
    }

    final_state = run_agent(state)

    print("FINAL RESULT:", final_state["result_path"])