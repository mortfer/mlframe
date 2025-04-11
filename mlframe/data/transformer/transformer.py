from abc import ABC, abstractmethod
from typing import Union, Callable, Dict, List
from mlframe.data.dataset import Dataset
import pandas as pd
import pyspark.sql as pys
from graphviz import Digraph
from copy import deepcopy, copy


class ITransformer(ABC):
    _instance_count = {}

    def __init__(self, inputs: list = None, outputs: list = None, name: str = None):
        self._inputs = inputs
        self._outputs = outputs if outputs is not None else inputs
        self._handle_name(name)

    @abstractmethod
    def transform(self, datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        try:
            # Handle input datasets
            # Core logic
            # Handle output datasets
            pass
        except Exception as e:
            self._handle_exception(e, context=" context")

    def __call__(self, dfs: Dict[str, Dataset]):
        if self._inputs is None:
            return self.transform(dfs)
        else:
            selected_dfs = {k: dfs[k] for k in self.inputs}
            results = self.transform(selected_dfs)
            not_selected_keys = set(dfs.keys()) - set(self.inputs)
            assert not set(not_selected_keys) & set(results.keys())
            outputs_dfs = {k: dfs[k] for k in not_selected_keys}
            outputs_dfs.update(results)
            return outputs_dfs

    def _handle_name(self, name):
        class_name = self.__class__.__name__
        if class_name not in self._instance_count:
            self._instance_count[class_name] = 0

        if name is None:
            self._instance_count[class_name] += 1
            self.name = f"{class_name}_{self._instance_count[class_name]}"
        else:
            self.name = name
        return self.name

    def _handle_exception(self, e, context=""):
        raise RuntimeError(f"'{self.name}' during {context} -> {str(e)} ") from e

    def plot_graph(self, dot_graph=None):
        pass


class FunctionTransformer(ITransformer):
    def __init__(
        self,
        functions: Union[Callable, List[Callable]],
        functions_input: str = "df",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.functions = functions if isinstance(functions, list) else [functions]
        self.functions_input = functions_input

    def transform(self, datasets: Dict[str, Dataset]):
        dict_keys = tuple(datasets.keys())
        assert len(dict_keys) == 1
        dataset = datasets[dict_keys[0]]
        if self.functions_input == " dataset":
            elem = dataset
        else:
            elem = getattr(dataset, self.functions_input)
        for i, func in enumerate(self.functions):
            try:
                elem = func(elem)
            except Exception as e:
                self._handle_exception(e, context=f"function {i}: {func.__name__}")
        if self.functions_input == "dataset":
            dataset = elem
        else:
            setattr(dataset, self.functions_input, elem)
        output_dict = {}
        if self._outputs is not None:
            output_dict[self._outputs[0]] = dataset
        else:
            output_dict[dict_keys[0]] = dataset
        return output_dict

    def plot_graph(self, dot_graph=None):
        if dot_graph is None:
            dot_graph = Digraph(name=f"{self.name}")
        label = f"""<
        <table border="0" cellborder="1" cellspacing="0" cellpadding="4">
        <tr><td><b>{self.name}()</b></td></tr>
        {"".join([f"<tr><td>{function}<td><tr>" for function in self.functions])}
        </table>
        >"""
        # dot.node(f"{self.name}", label=label, shape='box')
        return dot_graph


class SequentialTransformer(ITransformer):
    def __init__(self, transformers: List[ITransformer], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformers = transformers
        self.stopper = None

    def transform(self, dataset: Dict[str, Dataset]):
        for i, transformer in enumerate(self.transformers):
            try:
                if transformer.name == self.stopper or i == self.stopper:
                    print(
                        f">>[INFO] Stopping {self.name} jsut before {transformer.name}"
                    )
                    return dataset
                dataset = transformer(dataset)

            except Exception as e:
                self._hanlde_exception(e, context="SequentialTransformer")
        return dataset

    def plot_graph(self, dot_graph=None):
        if dot_graph is None:
            dot_graph = Digraph(
                name=f"{self.name}",
                comment="Data Transformation Pipeline",
                graph_attr={"rankdir": "LR"},
            )
        for transformer in self.transformers:
            dot_graph = transformer.plot_graph(dot_graph)

    def set_stopper(self, *, name: str = None, position: int = None):
        if name is not None:
            self.stopper = name
        else:
            self.stopper = position
        return self

    def _is_transformer_name_unique(self):
        # TODO
        return


class MultiTransformer(ITransformer):
    def __init__(
        self,
        transformers: Union[Dict[str, ITransformer], ITransformer],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.transformers = transformers

    def transform(self, dfs):
        result = {}
        if isinstance(self.transformers, dict):
            assert set(self.transformers.keys()) == set(dfs.keys())
            for name in self.transformers.keys():
                input_dict = {name: dfs[name]}
                output_dict = self.transformers[name](input_dict)
                result.update(output_dict)
        else:
            for name, df in dfs.items():
                input_dict = {name: df}
                output_dict = self.transformers(input_dict)
                result.append(output_dict)
        return result

    def plot_graph(self, dot_graph=None):
        if dot_graph is None:
            dot_graph = Digraph(name=f"{self.name}")
        return dot_graph


class EstimatorTransformer(ITransformer):
    def __init__(
        self,
        estimator_fn=None,
        df_to_fit: str = None,
        input_columns: List[str] = None,
        estimator_kwargs: Dict = None,
        estimator_loader: Callable = None,
        output_columns: List = None,
        transform_method: str = "transform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.estimator_fn = estimator_fn
        self.df_to_fit = df_to_fit
        self.estimator_kwargs = estimator_kwargs if estimator_kwargs is not None else {}
        self.input_columns = input_columns
        self.output_columns = (
            output_columns if output_columns is not None else input_columns
        )
        self.estimator_loader = estimator_loader
        self.transform_method = transform_method
        if self.estimator_loader is None:
            assert self.estimator_fn is not None

    def transform(self, datasets: Dict[str, Dataset]):
        dfs = {k: v.df for k, v in datasets.items()}
        # Fit
        if self.estimator_loader is not None:
            self.estimator_fn = self.estimator_loader()
        elif self.df_to_fit is not None:
            df = dfs[self.df_to_fit]
            if self.input_columns is not None:
                df = dfs[self.df_to_fit][self.input_columns]
            self.estimator_fn = self.estimator_fn.fit(df, **self.estimator_kwargs)
        # Transform
        for k, v in dfs.items():
            if self.input_columns is not None:
                v = v[self.input_columns]
            output_df = dfs[k]
            result = getattr(self.estimator_fn, self.transform_method)(v)

            if self.output_columns is not None:
                output_df[self.output_columns] = result
            else:
                output_df = result

            datasets[k].df = output_df
            datasets[k].estimators.append(self.estimator_fn)

        # Handle output
        output_dict = {}
        for input_name, output_name in zip(self.inputs, self._outputs):
            output_df[output_name] = datasets[input_name]
        return output_dict


class FeatureMetadataTransformer(ITransformer):
    def __init__(
        self,
        vocabularies: List = None,
        x: List = None,
        y: List = None,
        shape: bool = False,
        w=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocabularies = vocabularies if vocabularies is not None else []
        self.x = x
        self.y = y
        self.w = w
        self.shape = shape

    def get_vocabulary(self, df, feature_name):
        if isinstance(df, pd.DataFrame):
            vocab = tuple(sorted(df[feature_name].unique()))
        elif isinstance(df, pys.DataFrame):
            vocab = tuple(
                sorted(
                    df.select(feature_name).distinct().toPandas()[feature_name].values
                )
            )
        return vocab

    def get_shape(self, df):
        if isinstance(df, pd.DataFrame):
            shape = df.shape
        elif isinstance(df, pys.DataFrame):
            shape = (df.count(), len(df.columns))
        return shape

    def get_types(self, df):
        if isinstance(df, pd.DataFrame):
            types = dict(df.dtypes)
        elif isinstance(df, pys.DataFrame):
            types = None
        return types

    def transform(self, datasets: Dict[str, Dataset]):
        for name, dataset in datasets.items():
            if self.x is not None:
                dataset.x = self.x
            if self.y is not None:
                dataset.y = self.y
            if self.w is not None:
                dataset.w = self.w
            df = dataset.df
            original_metadata = dataset.metadata
            if len(self.vocabularies) > 0:
                features_vocabulary = {}
                for feature_name in self.vocabularies:
                    vocabulary = self.get_vocabulary(df, feature_name)
                    features_vocabulary[feature_name] = vocabulary
                original_metadata["vocabularies"] = features_vocabulary
            if self.shape is True:
                shape = self.get_shape(df)
                original_metadata["shape"] = shape
            original_metadata[" feature_types"] = self.get_types(df)
            dataset.metadata = original_metadata
        return datasets


class OneToNTransformer(ITransformer):
    def __init__(
        self, core_function, input_elements, dataset_output_mode: str = "copy", **kwargs
    ):
        super().__init__(**kwargs)
        self.core_function = core_function
        self.input_elements = input_elements
        self.dataset_output_mode = dataset_output_mode

    def transform(self, datasets: Dict[str, Dataset]):
        dict_keys = list(datasets.keys())
        assert len(dict_keys) == 1
        inputs = [getattr(datasets[dict_keys[0]], elem) for elem in self.input_elemnts]
        results = self.core_function(*inputs)

        outputs_dict = {}
        if self.dataset_output_mode == "copy":
            for i, name in enumerate(self._outputs):
                dataset_attrs = {
                    key: copy(value)
                    for key, value in datasets[dict_keys[0]].__dict__.items()
                    if key != "df"
                }
                dataset = Dataset(df=None, **dataset_attrs)
                outputs_dict[name] = dataset
        elif self.dataset_output_mode == "new":
            for i, name in enumerate(self._outputs):
                dataset = Dataset()
                outputs_dict[name] = dataset
        else:
            raise ValueError()

        for i, name in enumerate(self._outputs):
            outputs_dict[name].df = results[i]

        return outputs_dict


class TwoToOneTransformer(ITransformer):
    def __init__(
        self,
        merge_function: Callable,
        auxiliar_reduction: Callable = None,
        bypass_auxiliar: bool = False,
        auxiliar_kwarg_injection: Callable = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.merge_function = merge_function
        self.bypass_auxiliar = bypass_auxiliar
        self.auxiliar_kwarg_injection = auxiliar_kwarg_injection

    def transform(self, datasets: Dict[str, Dataset]):
        auxiliar_dataset = datasets[self.inputs[1]]
        main_dataset = datasets[self.inputs[0]]
        auxiliar_result = auxiliar_dataset.df
        if self.auxiliar_kwarg_injection is not None:
            auxiliar_result = self.auxiliar_kwarg_injection(auxiliar_result)
            df = self.merge_function(main_dataset.df, **auxiliar_result)
        else:
            df = self.merge_function(main_dataset.df, auxiliar_result)

        outputs_dict = {}
        output_name = self._outputs[0]
        main_dataset.df = df
        outputs_dict[output_name] = main_dataset
        if self.bypass_auxiliar is True:
            outputs_dict[self._outputs[1]] = auxiliar_dataset
        return outputs_dict


class DataIteratorTransformer(ITransformer):
    def __init__(
        self,
        batch_size: int,
        shuffle: bool,
        generator_fn: Callable,
        epochs: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator_fn = generator_fn
        self.epochs = epochs

    def transform(self, datasets: Dict[str, Dataset]):
        for name, dataset in datasets.items():
            try:
                df = dataset.df
                feature_names = dataset.x
                y_name = dataset.y
                data_iterator = self.generator_fn(
                    df,
                    batch_size=self.batch_size,
                    feature_names=feature_names,
                    y_name=y_name,
                    shuffle=self.shuffle,
                    epochs=self.epochs,
                )
                dataset.data_iterator = data_iterator
                dataset.metadata["batch_size"] = self.batch_size
                dataset.metadata["shuffle"] = self.shuffle
                dataset.metadata["epochs"] = self.epochs
            except Exception as e:
                self._handle_exception(
                    e, context=f"dataiterator converter with dataset {name}"
                )
        return datasets
