from mlflow.tracking import MlflowClient

def stage_model(model_name: str, version: int):
    client = MlflowClient()

    # 1️⃣ Get all versions of the model
    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    # 2️⃣ Archive existing Staging models
    for v in versions:
        if v.current_stage == "Staging":
            client.transition_model_version_stage(
                name=model_name,
                version=v.version,
                stage="Archived"
            )
            print(
                f"📦 Archived old Staging model: "
                f"{model_name} v{v.version}"
            )

    # 3️⃣ Promote new model to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging"
    )

    print(
        f"🚀 Model {model_name} v{version} "
        f"moved to Staging successfully"
    )
