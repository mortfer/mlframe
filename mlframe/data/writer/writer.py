from abc import ABC, abstractmethod
from typing import Union, Dict, List
from mlframe.data.dataset import Dataset

class IWriter(ABC):
    def __init__(self, inputs: Union[List, str] = None):
        self.inputs = inputs

    @abstractmethod
    def write(self):
        pass

    def __call__(self, dfs: Union[Dataset, Dict[str, Dataset]]):
        if self.inputs is None:
            return self.write(dfs)
            
        if isinstance(self.inputs, str):
            selected_dfs = dfs[self.inputs]
        else:
            selected_dfs = {k: dfs[k] for k in self.inputs}
        
        results = self.write(selected_dfs)
        
        if not isinstance(results, dict):
            results = {self.inputs: results}
            
        not_selected_keys = set(dfs.keys()) - set(self.inputs)
        output_dfs = {k: dfs[k] for k in not_selected_keys}
        for name, result in results.items():
            output_dfs[name] = result
                
        return output_dfs

class SparkWriter(IWriter):
    def __init__(self, table_name: str, source: str = "parquet", mode: str = "error", **kwargs):
        """
        Args:
            table_name: String name of the data source, e.g. 'json', 'parquet'.
        """
        super().__init__(**kwargs)
        self.source = source
        self.mode = mode
        self.table_name = table_name

    def write(self, dataset: Dataset):
        df = dataset.df
        df.write.format(self.source).mode(self.mode).saveAsTable(self.table_name)
        return dataset

class MultiWriter(IWriter):
    """Multiple writers for one dataset"""
    def __init__(self, writers: List[IWriter], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writers = writers

    def write(self, dataset: Dataset):
        for writer in self.writers:
            writer(dataset)
        return dataset

class SequentialWriter(IWriter):
    """Orchestrator between datasets and writers"""
    def __init__(self, writers: Dict[str, IWriter], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writers = writers

    def write(self, datasets: Dict[str, Dataset]):
        result = {}
        for key in list(datasets.keys()):
            dataset = self.writers[key](datasets[key])
            result[key] = dataset
        return result

