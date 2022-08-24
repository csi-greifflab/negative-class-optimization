import numpy as np
import mlflow

EXPERIMENT_ID = 1

with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="0x"):
    
    
    print(mlflow.get_artifact_uri())
    run_id = mlflow.active_run().info.run_id
    

    mlflow.log_param("param_const", 10)
    mlflow.log_param("param_rand", np.random.random(1))
    mlflow.log_metric("m1", np.random.random(1))

    # with open("test_artifact.txt", "w+") as fh:
    #     fh.write("Some dummy_output")
    # mlflow.log_artifact(
    #     "./test_artifact.txt",
    #     "backend_stor"
    # )
    
    mlflow.set_tag("purpose", "test")
