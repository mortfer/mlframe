from dataclasses import dataclass, is_dataclass, fields
from functools import wraps
from mlframe.data.dataset_pipeline import DatasetPipeline
from mlframe.training.trainer import ITrainer

@dataclass(frozen=True)
class ExperimentRun:
    """Class for keeping track of an experiment composed by training, prediction and evaluation steps"""
    run_name: str
    experiment_name: str
    training_dataset_pipeline: DatasetPipeline
    trainer: ITrainer
    validation_dataset_pipeline: DatasetPipeline = None
    testing_dataset_pipeline: DatasetPipeline = None
    predictor = None
    evaluator = None

@dataclass(frozen=True)
class TableRun:
    """Class for keeping track of a dataset pipeline that typically saves the result somewhere"""
    name: str
    dataset_pipeline: DatasetPipeline

def parse_dataclass(fn):
    @wraps(fn)
    def wrapper(first_arg=None, *args, **kwargs):
        # Check if the first argument is a dataclass
        if first_arg and is_dataclass(first_arg):
            # Merge dataclass attributes with additional kwargs
            dataclass_fields = {
                field.name: getattr(first_arg, field.name)
                for field in fields(first_arg)
            }
            kwargs.update(dataclass_fields)
        elif first_arg:
            # If not a dataclass, include it as a positional argument
            args = (first_arg,) + args
        return fn(*args, **kwargs)
    return wrapper