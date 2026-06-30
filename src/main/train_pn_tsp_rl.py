from src.main.train_common import run_pipeline


if __name__ == "__main__":
    run_pipeline(model_name="pn", problem="tsp", mode="rl")
