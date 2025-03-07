from mlframe.dataclasses import parse_dataclass


@parse_dataclass
def main_train(
    run_name: str,
    experiment_name: str,
    trainer,
    training_dataset_pipeline,
    validation_dataset_pipeline,
    force_run: bool = False,
    **kwargs,
):
    result = trainer(
        run_name=run_name,
        experiment_name=experiment_name,
        training_dataset_pipeline=training_dataset_pipeline,
        validation_dataset_pipeline=validation_dataset_pipeline,
        force_run=force_run,
        **kwargs,
    )

    return result


@parse_dataclass
def main_evaluate(
    run_name: str, experiment_name: str, evaluator, testing_dataset_pipeline, **kwargs
):
    result = evaluator(
        run_name=run_name,
        experiment_name=experiment_name,
        testing_dataset_pipeline=testing_dataset_pipeline,
        **kwargs,
    )

    return result
