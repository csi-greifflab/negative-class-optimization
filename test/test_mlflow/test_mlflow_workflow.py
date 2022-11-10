import numpy as np
import matplotlib.pyplot as plt
import mlflow

TEST_EXPERIMENT_NAME = f"TEST"

mlflow.set_tracking_uri("http://0.0.0.0:5000")
experiment = mlflow.set_experiment(TEST_EXPERIMENT_NAME)

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="0x", description="random description"):
    
    run_id = mlflow.active_run().info.run_id

    # mlflow.log_param("param_const", 10)
    # mlflow.log_param("param_rand", np.random.random(1))
    mlflow.log_params({
        "param_const": 10,
        "param_rand": np.random.random(1),
    })

    # mlflow.log_metric("m1", np.random.random(1), step=1)
    mlflow.log_metrics({
        "m1": np.random.random(1)[0],
        "m2": np.random.random(1)[0],
    }, step=1)
    mlflow.log_metric("m1", np.random.random(1), step=2)

    mlflow.log_dict({"some_key": "some_value"}, "test_dict.json")

    fig, ax = plt.subplots()
    ax.plot([0, 1], [2, 3])
    mlflow.log_figure(fig, "test_figure.png")

    mlflow.log_text("More dummy text", "test_txt.txt")

    with open("test_artifact.txt", "w+") as fh:
        fh.write("Some dummy_output")
    mlflow.log_artifact(
        "./test_artifact.txt"
    )
    
    mlflow.set_tag("purpose", "test")
