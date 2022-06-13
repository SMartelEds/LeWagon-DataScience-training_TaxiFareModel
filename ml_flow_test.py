import mlflow
from mlflow.tracking import MlflowClient
EXPERIMENT_NAME = "test_experiment"

# Comment this to go local
mlflow.set_tracking_uri("https://mlflow.lewagon.ai/")

client = MlflowClient()
try:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
except BaseException:
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

# Comment these to go local
yourname = 'SMartelEds'
if yourname is None:
    print("please define your name, il will be used as a parameter to log")


for model in ["linear", "Randomforest"]:
    run = client.create_run(experiment_id)
    client.log_metric(run.info.run_id, "rmse", 4.5)
    client.log_param(run.info.run_id, "model", model)
    # Comment this to go local
    client.log_param(run.info.run_id, "student_name", yourname)
