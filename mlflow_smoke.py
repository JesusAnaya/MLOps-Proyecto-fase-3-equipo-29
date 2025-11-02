import os, mlflow
print("URI=", os.environ.get("MLFLOW_TRACKING_URI"))
print("EXP=", os.environ.get("MLFLOW_EXPERIMENT"))
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT","equipo-29"))
with mlflow.start_run(run_name="smoke_test_params"):
    mlflow.log_param("ping","pong")
    mlflow.log_metric("hello_metric", 1.0)
print("OK")
