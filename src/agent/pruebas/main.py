# main.py
from graph import run_agent

if __name__ == "__main__":

    state = {
        "image_path": "src/cambio_fondo/img.png",
        "prompt": "Pon esta botella en un bosque humedo al atardecer"
    }

    final_state = run_agent(state)

    print("FINAL RESULT:", final_state["result_path"])