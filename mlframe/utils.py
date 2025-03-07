import os
import mlflow
import warnings


class MLFlowHandler:
    DATABRICKS_URI = os.path.join(
        "dbfs:",
        "databricks",
        "mlflow-tracking",
        "{experiment_id}",
        "{run_id}",
        "artifacts",
    )

    def __init__(
        self, run_name: str, experiment_name: str, nested: bool = False, **kwargs
    ):
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.run = None
        self.run_id = None
        self.experiment_id = None
        self.nested = nested
        self.str_run = "child run" if nested else "parent run"

    def start_run(
        self,
    ):
        pass

    def end_run(
        self,
    ):
        pass

    def model_from_active_run(
        self,
    ):
        pass

    def __call__(
        self,
    ):
        pass

    @classmethod
    def model_from_mlflow(
        cls,
    ):
        pass
