from abc import ABC, abstractmethod
import mlflow
from io import StringIO
from mlframe.data import Dataset

class ITrainingDatasetLoader(ABC):
    @abstractmethod
    def load(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.load(*args, **kwargs)

class TrainingDatasetLoader(ITrainingDatasetLoader):
    def __init__(self, training_key: str="training_df", validation_key="validation_df", log: bool = True):
        self.training_key = training_key
        self.validation_key = validation_key
        self.log = log

    def __call__(self, training_dataset_pipeline, validation_dataset_pipeline=None, **experiment):
        if validation_dataset_pipeline is None:
            datasets = training_dataset_pipeline()
            training_dataset = datasets[self.training_key]
            validation_dataset = datasets[self.validation_key]
        else:
            training_dataset = training_dataset_pipeline()[self.training_key]
            validation_dataset = validation_dataset_pipeline()[self.validation_key]
        if self.log is True:
            print("log_validation_dataset", validation_dataset)
        return training_dataset, validation_dataset
    
def log_training_metadata(dataset: Dataset):
    df = dataset.df
    if isinstance(df, pd.DataFrame):
        for part, df in [("head", df.head(10)), ("tail", df.tail(10))]:
            buffer = StringIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            mlflow.log_text(buffer.getvalue(), artifact_value=f"tables/training_df_{part}.csv")
    shape = dataset.metadata.get("shape", None)
    if shape is not None:
        mlflow.log_param("training_df_shape", shape)
    if dataset.x is not None:
        mlflow.log_param("x", dataset.x)
    if dataset.y is not None:
        mlflow.log_param("y", dataset.y)
    return None

def log_validation_metadata(dataset: Dataset):
    df = dataset.df
    if isinstance(df, pd.DataFrame):
        for part, df in [("head", df.head(10)), ("tail", df.tail(10))]:
            buffer = StringIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            mlflow.log_text(buffer.getvalue(), artifact_value=f"tables/validation_df_{part}.csv")
    shape = dataset.metadata.get("shape", None)
    if shape is not None:
        mlflow.log_param("validation_df_shape", shape)
    return None
