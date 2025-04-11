from abc import ABC, abstractmethod
from typing import Dict
from mlframe.data.dataset import Dataset
from mlframe.data.utils import SparkSessionManager


class IReader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def read(self):
        pass

    def __call__(self):
        return self.read()


class SparkReader(IReader):
    def __init__(self, path, mode="delta_table", **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.mode = mode

    def read(self):
        spark = SparkSessionManager.get_spark()
        if self.mode == "delta_table":
            return Dataset(df=spark.table(self.path))
        elif self.mode == "full_path":
            return Dataset(df=spark.read.parquet(self.path))


class SnowflakeReader(IReader):
    def __init__(self, query, **kwargs):
        super().__init__(**kwargs)
        self.query = query

    def read(self):
        # TODO
        # df = snowflakeconnector(self.query)
        # return Dataset(df=df)
        pass


class SequentialReader(IReader):
    def __init__(self, readers: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.readers = readers

    def read(self):
        datasets = {}
        for name, reader in self.readers.items():
            datasets[name] = reader()
        return datasets


class BypassReader(IReader):
    def __init__(self, df, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = df

    def read(self):
        return Dataset(df=self.df)
