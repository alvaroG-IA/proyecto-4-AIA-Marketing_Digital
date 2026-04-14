# main.py
from graph import run_agent

if __name__ == "__main__":

    state = {
        "image_path": "../../cambio_fondo/img.png",
        "prompt": "put the bottle in a cyberpunk street at night"
    }

    final_state = run_agent(state)

    print("FINAL RESULT:", final_state["result_path"])