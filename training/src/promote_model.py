# training/src/promote_model.py

import mlflow
from mlflow.tracking import MlflowClient
from common.config import MODEL_NAME, MLFLOW_TRACKING_URI


def get_accuracy(client, version):
    run_id = client.get_model_version(
        MODEL_NAME, version
    ).run_id
    run = mlflow.get_run(run_id)
    return run.data.metrics.get("accuracy", 0)


def rotate_model_lifecycle(new_version: int):

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print("\n🔄 Starting Smart Model Governance...")

    # --------------------------------------------------
    # Get existing aliases
    # --------------------------------------------------
    try:
        old_candidate = client.get_model_version_by_alias(
            MODEL_NAME, "candidate"
        ).version
    except Exception:
        old_candidate = None

    try:
        old_production = client.get_model_version_by_alias(
            MODEL_NAME, "production"
        ).version
    except Exception:
        old_production = None

    # --------------------------------------------------
    # Get accuracies
    # --------------------------------------------------
    new_acc = get_accuracy(client, new_version)
    candidate_acc = get_accuracy(client, old_candidate) if old_candidate else 0
    production_acc = get_accuracy(client, old_production) if old_production else 0

    print(f"New Accuracy: {new_acc}")
    print(f"Candidate Accuracy: {candidate_acc}")
    print(f"Production Accuracy: {production_acc}")

    # --------------------------------------------------
    # Step 1: Decide Candidate Update
    # --------------------------------------------------
    if new_acc > candidate_acc:
        print("🟢 New model better than Candidate → updating candidate")

        # Archive old candidate
        if old_candidate:
            client.delete_registered_model_alias(
                MODEL_NAME, "candidate"
            )
            client.set_registered_model_alias(
                MODEL_NAME, "previous_candidate", old_candidate
            )

        # Set new as candidate
        client.set_registered_model_alias(
            MODEL_NAME, "candidate", new_version
        )

    else:
        print("🔴 New model not better than Candidate → ignoring")
        return

    # --------------------------------------------------
    # Step 2: Decide Production Promotion
    # --------------------------------------------------
    if new_acc > production_acc:
        print("🚀 Candidate better than Production → promoting")

        # Archive old production
        if old_production:
            client.delete_registered_model_alias(
                MODEL_NAME, "production"
            )

        client.set_registered_model_alias(
            MODEL_NAME, "production", new_version
        )

    print("✅ Smart governance completed.\n")
