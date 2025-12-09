# training/src/stage_model.py
from mlflow.tracking import MlflowClient

def stage_model(model_name, version):
    client = MlflowClient()

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=True
    )

    print(f"Model {model_name} version {version} moved to STAGING")
