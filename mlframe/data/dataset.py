from typing import Union, Dict, Iterable, List


class Dataset:
    def __init__(
        self,
        df=None,
        data_iterator: Iterable = None,
        estimators: List = None,
        metadata: Dict = None,
        x: List = None,
        y: List = None,
        w: List = None,
    ):
        self.df = df
        self.data_iterator = data_iterator
        self.metadata = metadata if metadata is not None else {}
        self.estimators = estimators if estimators is not None else []
        self.x = x
        self.y = y
        self.w = w
