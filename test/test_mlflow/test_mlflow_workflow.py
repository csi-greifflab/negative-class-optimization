import numpy as np
import mlflow

TEST_EXPERIMENT_NAME = f"TEST"


mlflow.set_tracking_uri("http://0.0.0.0:5000")
experiment = mlflow.set_experiment(TEST_EXPERIMENT_NAME)

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="0x"):
    
    print(mlflow.get_artifact_uri())
    run_id = mlflow.active_run().info.run_id

    mlflow.log_param("param_const", 10)
    mlflow.log_param("param_rand", np.random.random(1))
    mlflow.log_metric("m1", np.random.random(1))

    with open("test_artifact.txt", "w+") as fh:
        fh.write("Some dummy_output")
    mlflow.log_artifact(
        "./test_artifact.txt"
    )
    
    mlflow.set_tag("purpose", "test")
