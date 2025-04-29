from typing import Dict, List, Union
from mlframe.training.utils import CallbackFactory
from mlframe.training.utils import MetricFactory
import keras
import gc
import math
from abc import ABC, abstractmethod
from contextlib import ExitStack

class ITrainingLoop(ABC):
    @abstractmethod
    def train(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.train(*args, **kwargs)

class KerasTrainingLoop(ITrainingLoop):
    def __init__(
        self,
        model_compile: Dict = None,
        steps_per_epoch: int = None,
        dataset_element: str = 'df',
        validation_steps: int = None,
        callbacks: Union[List, str] = None,
        context_iterator: bool = False,
        dimensionality_reduction: bool = False,
        **kwargs
    ):
        self.model_compile = model_compile
        self.steps_per_epoch = steps_per_epoch
        self.dataset_element = dataset_element
        self.validation_steps = validation_steps
        self.callbacks = callbacks
        self.context_iterator = context_iterator
        self.dimensionality_reduction = dimensionality_reduction
        self.kwargs = kwargs
        assert self.dataset_element in ("df", "data_iterator")

    def train(self, model, training_dataset, validation_dataset, **kwargs):
        keras.backend.clear_session()
        gc.collect()
        if self.model_compile is not None:
            if isinstance(self.model_compile.get("metrics"), str):
                metrics = MetricFactory.get(self.model_compile["metrics"])
                self.model_compile["metrics"] = metrics
            model.compile(**self.model_compile)

        training_data = getattr(training_dataset, self.dataset_element)
        validation_data = getattr(validation_dataset, self.dataset_element)

        if self.steps_per_epoch is None and self.dataset_element == "data_iterator":
            training_batch_size = training_dataset.metadata["batch_size"]
            self.steps_per_epoch = math.ceil(training_dataset.metadata["shape"][0] / training_batch_size)
        if self.validation_steps is None and self.dataset_element == "data_iterator":
            validation_batch_size = validation_dataset.metadata["batch_size"]
            self.validation_steps = math.ceil(validation_dataset.metadata["shape"][0] / validation_batch_size)

        if isinstance(self.callbacks, str):
            callbacks = CallbackFactory.get(self.callbacks)
        else:
            callbacks = self.callbacks


        with ExitStack() as stack:
            if self.context_iterator is True:
                training_data = stack.enter_context(training_data)
                validation_data = stack.enter_context(validation_data)
                print("[INFO] >> Starting model fitting")
            if self.dimensionality_reduction is True:
                history = model.fit(
                    training_data,
                    training_data,
                    validation_data=(validation_data, validation_data),
                    steps_per_epoch=self.steps_per_epoch,
                    validation_steps=self.validation_steps,
                    callbacks=callbacks,
                    **kwargs
                )
            else:
                history = model.fit(
                    training_data,
                    validation_data=validation_data,
                    steps_per_epoch=self.steps_per_epoch,
                    validation_steps=self.validation_steps,
                    callbacks=callbacks,
                    **self.kwargs
                )
        return model, history
