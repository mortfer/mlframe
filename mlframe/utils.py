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

    def start_run(self, force_run=False, **kwargs):
        if self.nested is False:
            if mlflow.active_run() is not None:
                warnings.warn("Another active run found, ending it")
                mlflow.end_run()
            mlflow.set_experiment(self.experiment_name)
            mlflow_experiment = mlflow.get_experiment_by_name(self.experiment_name)
            runs = mlflow.search_runs(
                experiment_ids=mlflow_experiment.experiment_id,
                filter_string=f'tags.mlflow.runName = "{self.run_name}"',
            )
            if "tags.mlflow.parentRunId" not in runs.columns:
                runs["tags.mlflow.parentRunId"] = None
            runs = runs[runs["tags.mlflow.parentRunId"].isnull()]
            run_ids = runs.run_id
        else:
            mlflow.set_experiment(self.experiment_name)
            mlflow_experiment = mlflow.get_experiment_by_name(self.experiment_name)
            run_ids = mlflow.search_runs(
                experiment_ids=mlflow_experiment.experiment_id,
                filter_string=f'tags.mlflow.runName = "{self.run_name}" and tags.mlflow.parentRunId = "{mlflow.active_run().info_run_id}"',
            ).run_id

            if len(run_ids) == 0:
                run_id = None
                run_name = self.run_name
                print(f">>[INFO] Starting new {self.str_run}")
            else:
                if force_run:
                    run_id = run_ids[0]
                    run_name = None
                    print(f">>[INFO] Run already exists. Resuming {self.str_run}...")
                else:
                    raise Exception(
                        "Run found. Set force_run=True if you want to resume it"
                    )
            self.run = mlflow.start_run(
                experiment_id=mlflow_experiment.experiment_id,
                run_id=run_id,
                run_name=run_name,
                nested=self.nested,
            )
            self.run_id = self.run.info.run_id
            self.experiment_id = mlflow_experiment.experiment_id
            return self.run

    def end_run(
        self,
    ):
        if self.run:
            print(f">>[INFO] {self.str_run.capitalize()} ended.")
            mlflow.end_run()
            self.run = None
        else:
            print(" No active MLFlow run")

    def model_from_active_run(
        self,
    ):
        pass

    def __call__(self, *args, **kwargs):
        self.start_run(*args, **kwargs)
        return self

    def __enter__(self):
        return self.run

    def __exit__(self):
        self.end_run()

    def model_from_active_run(
        self,
        relative_path: str = None,
        databricks: bool = True,
        flavor: str = " tensorflow",
        **kwargs,
    ):
        # kwargs like keras_model_kwargs={"compile":False}
        if databricks:
            model_uri = self.DATABRICKS_URI.format(
                experiment_id=self.experiment_id, run_id=self.run_id
            )
        else:
            model_uri = None
            raise ValueError("MLFlowHandler does not have model_uri")
        if relative_path:
            model_uri = os.path.join(model_uri, relative_path)
        flavor_method = getattr(mlflow, flavor)
        try:
            model = flavor_method.load_model(model_uri, **kwargs)
        except Exception as error:
            print(error)
            warnings.warn(f"Model not found in {model_uri}")
            model = None
        return model

    @classmethod
    def model_from_mlflow(
        cls,
        run_name,
        experiment_name,
        relative_path: str = None,
        databricks: bool = True,
        flavor: str = " tensorflow",
        **kwargs,
    ):
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        run_ids = mlflow.search_runs(
            experiment_ids=experiment_id,
            filter_string=f'tags.mlflow.runName = "{run_name}"',
        ).run_id

        if len(run_ids) == 0:
            raise Exception(f"Run with {run_name=} and {experiment_name=} not found")
        else:
            run_id = run_ids[0]
        if databricks:
            model_uri = cls.DATABRICKS_URI.format(
                experiment_id=experiment_id, run_id=run_id
            )
        else:
            model_uri = None
            raise ValueError("MLFlowHandler does not have model_uri")
        if relative_path:
            model_uri = os.path.join(model_uri, relative_path)
        flavor_method = getattr(mlflow, flavor)
        try:
            model = flavor_method.load_model(model_uri, **kwargs)
        except Exception as error:
            print(error)
            warnings.warn(f"Model not found in {model_uri}")
            model = None
        return model

    @classmethod
    def load_dict(
        cls,
        run_name,
        experiment_name,
        relative_path: str = None,
        databricks: bool = True,
        **kwargs,
    ):
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        run_ids = mlflow.search_runs(
            experiment_ids=experiment_id,
            filter_string=f'tags.mlflow.runName = "{run_name}"',
        ).run_id

        if len(run_ids) == 0:
            raise Exception(f"Run with {run_name=} and {experiment_name=} not found")
        else:
            run_id = run_ids[0]
        if databricks:
            model_uri = cls.DATABRICKS_URI.format(
                experiment_id=experiment_id, run_id=run_id
            )
        else:
            model_uri = None
            raise ValueError("MLFlowHandler does not have model_uri")
        if relative_path:
            model_uri = os.path.join(model_uri, relative_path)
        dictionary = mlflow.artifacts.load_dict(model_uri, **kwargs)
        return dictionary


class MLFlowHandlerNew:
    DATABRICKS_URI = os.path.join(
        "dbfs:",
        "databricks",
        "mlflow-tracking",
        "{experiment_id}",
        "{run_id}",
        "artifacts",
    )

    def __init__(
        self, run_name: str, experiment_name: str, parent_run_name: str = None, **kwargs
    ):
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.parent_run_name = parent_run_name
        self.experiment_id = None
        self.run = None
        self.run_id = None
        self.parent_run = None
        self.parent_run_id = None

    @staticmethod
    def _get_run(run_name, experiment_id, force_run):
        runs = mlflow.search_runs(
            experiment_ids=experiment_id,
            filter_string=f'tags.mlflow.runName = "{run_name}"',
        )
        if "tags.mlflow.parentRunId" not in runs.columns:
            runs["tags.mlflow.parentRunId"] = None
        runs = runs[runs["tags.mlflow.parentRunId"].isnull()]
        run_ids = runs.run_id
        if len(run_ids) == 0:
            run_id = None
            run_name = run_name
            print(f">>[INFO] Starting new run")
        else:
            if force_run:
                run_id = run_ids[0]
                run_name = None
                print(f">>[INFO] Run already exists. Resuming run...")
            else:
                raise Exception(
                    "Run found. Set force_run=True if you want to resume it"
                )
        run = mlflow.start_run(
            experiment_id=experiment_id,
            run_id=run_id,
            run_name=run_name,
            nested=False,
        )
        return run

    def start_run(self, force_run=False, **kwargs):
        mlflow.set_experiment(self.experiment_name)
        mlflow_experiment = mlflow.get_experiment_by_name(self.experiment_name)
        self.experiment_id = mlflow_experiment.experiment_id

        if self.parent_run_name is None:
            if mlflow.active_run() is not None:
                warnings.warn("Another active run found, ending it")
                mlflow.end_run()

            self.run = self._get_run(self.run_name, self.experiment_id, force_run)
            self.run_id = self.run.info.run_id

        else:
            self.parent_run = self._get_run(
                self.parent_run_name, self.experiment_id, force_run=True
            )

            child_run_ids = mlflow.search_runs(
                experiment_ids=self.experiment_id,
                filter_string=f'tags.mlflow.runName = "{self.run_name}" and tags.mlflow.parentRunId = "{self.parent_run.info_run_id}"',
            ).run_id

            if len(child_run_ids) == 0:
                run_id = None
                run_name = self.run_name
                print(f">>[INFO] Starting new child run")
            else:
                if force_run:
                    run_id = child_run_ids[0]
                    run_name = None
                    print(f">>[INFO] Run already exists. Resuming child run...")
                else:
                    raise Exception(
                        "Run found. Set force_run=True if you want to resume it"
                    )
            self.run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_id=run_id,
                run_name=run_name,
                nested=True,
            )
            self.run_id = self.run.info.run_id

        return self.run

    def end_run(
        self,
    ):
        if self.run:
            print(f">>[INFO] {self.str_run.capitalize()} ended.")
            mlflow.end_run()
            # TODO: Handle nested run
            self.run = None
        else:
            print(" No active MLFlow run")

    def model_from_active_run(
        self,
    ):
        pass

    def __call__(self, *args, **kwargs):
        self.start_run(*args, **kwargs)
        return self

    def __enter__(self):
        return self.run

    def __exit__(self):
        self.end_run()

    def model_from_active_run(
        self,
        relative_path: str = None,
        databricks: bool = True,
        flavor: str = " tensorflow",
        **kwargs,
    ):
        # kwargs like keras_model_kwargs={"compile":False}
        if databricks:
            model_uri = self.DATABRICKS_URI.format(
                experiment_id=self.experiment_id, run_id=self.run_id
            )
        else:
            model_uri = None
            raise ValueError("MLFlowHandler does not have model_uri")
        if relative_path:
            model_uri = os.path.join(model_uri, relative_path)
        flavor_method = getattr(mlflow, flavor)
        try:
            model = flavor_method.load_model(model_uri, **kwargs)
        except Exception as error:
            print(error)
            warnings.warn(f"Model not found in {model_uri}")
            model = None
        return model

    @classmethod
    def model_from_mlflow(
        cls,
        run_name,
        experiment_name,
        relative_path: str = None,
        databricks: bool = True,
        flavor: str = " tensorflow",
        **kwargs,
    ):
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        run_ids = mlflow.search_runs(
            experiment_ids=experiment_id,
            filter_string=f'tags.mlflow.runName = "{run_name}"',
        ).run_id

        if len(run_ids) == 0:
            raise Exception(f"Run with {run_name=} and {experiment_name=} not found")
        else:
            run_id = run_ids[0]
        if databricks:
            model_uri = cls.DATABRICKS_URI.format(
                experiment_id=experiment_id, run_id=run_id
            )
        else:
            model_uri = None
            raise ValueError("MLFlowHandler does not have model_uri")
        if relative_path:
            model_uri = os.path.join(model_uri, relative_path)
        flavor_method = getattr(mlflow, flavor)
        try:
            model = flavor_method.load_model(model_uri, **kwargs)
        except Exception as error:
            print(error)
            warnings.warn(f"Model not found in {model_uri}")
            model = None
        return model

    @classmethod
    def load_dict(
        cls,
        run_name,
        experiment_name,
        relative_path: str = None,
        databricks: bool = True,
        **kwargs,
    ):
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        run_ids = mlflow.search_runs(
            experiment_ids=experiment_id,
            filter_string=f'tags.mlflow.runName = "{run_name}"',
        ).run_id

        if len(run_ids) == 0:
            raise Exception(f"Run with {run_name=} and {experiment_name=} not found")
        else:
            run_id = run_ids[0]
        if databricks:
            model_uri = cls.DATABRICKS_URI.format(
                experiment_id=experiment_id, run_id=run_id
            )
        else:
            model_uri = None
            raise ValueError("MLFlowHandler does not have model_uri")
        if relative_path:
            model_uri = os.path.join(model_uri, relative_path)
        dictionary = mlflow.artifacts.load_dict(model_uri, **kwargs)
        return dictionary
