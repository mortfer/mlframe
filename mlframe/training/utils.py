from typing import Union, Protocol, Callable, Dict, List, Any, Tuple
import numbers
import math
import keras
from keras.callbacks import Callback, EarlyStopping
from keras.metrics import Metric
from mlframe.training.callbacks import LRCallback, MLFlowCallback, MLFlowModelCheckpoint
from mlframe.models.keras.metrics import WAPE, Bias, RelativeBias

class CallbackFactory:
    """Ensures that a new instance of the callback is created every time the factory provides callbacks"""
    _default_registry: Dict[str, List[Callable[[], Callback]]] = {
        "mlflow_basic": [
            lambda: LRCallback(),
            lambda: MLFlowCallback(),
            lambda: MLFlowModelCheckpoint("models", monitor="val_loss", save_best_only=True,mode="min"),
            lambda: EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ],
    }

    _user_registry: Dict[str, List[Callable[[], Callback]]] = {}

    @classmethod
    def register(cls, name: str, callbacks: List[Callable[[], Callback]]) -> None:
        """Register user callbacks"""
        cls._user_registry[name] = callbacks

    @classmethod
    def get(cls, name: str, **kwargs) -> List[Callback]:
        """Merge multiple callback sets from registry"""
        if name in cls._user_registry:
            return [callback(**kwargs) if callable(callback) else callback for callback in cls._user_registry[name]]
        if name in cls._default_registry:
            return [callback(**kwargs) if callable(callback) else callback for callback in cls._default_registry[name]]
        raise ValueError(f"Callback set '{name}' is not registered.")

    @classmethod
    def available_callbacks(cls) -> Dict[str, List[str]]:
        """Return all registered callback sets (default and user-defined)"""
        return {
            "default": list(cls._default_registry.keys()),
            "user": list(cls._user_registry.keys())
        }

class MetricFactory:
    """Factory class for managing and creating metric instances"""
    _default_registry: Dict[str, List[Callable[[], Metric]]] = {
        "basic": [
            lambda: keras.metrics.MeanSquaredLogarithmicError(),
            lambda: keras.metrics.MeanAbsoluteError(),
            lambda: Bias(),
            lambda: WAPE(),
            lambda: RelativeBias(epsilon=1),
        ]
    }

    _user_registry: Dict[str, List[Callable[[], Metric]]] = {}

    @classmethod
    def register(cls, name: str, metrics: List[Callable[[], Metric]]) -> None:
        """Register a list of metrics to the user registry"""
        cls._user_registry[name] = metrics

    @classmethod
    def get(cls, name: str, **kwargs) -> List[Metric]:
        """Get metrics by name from registry over default registry"""
        if name in cls._user_registry:
            return [metric(**kwargs) if callable(metric) else metric for metric in cls._user_registry[name]]
        if name in cls._default_registry:
            return [metric(**kwargs) if callable(metric) else metric for metric in cls._default_registry[name]]
        raise ValueError(f"Metric set '{name}' is not registered.")

    @classmethod
    def available_metrics(cls) -> Dict[str, List[str]]:
        """Return all registered metric sets (default and user-defined)"""
        return {
            "default": list(cls._default_registry.keys()),
            "user": list(cls._user_registry.keys())
        }

class OptunaState:
    _params = {}
    
    @classmethod
    def set(cls, qualified_name, mutable_param):
        if qualified_name in cls._params:
            raise ValueError(f"Key {qualified_name} already exists in OptunaState")
        cls._params[qualified_name] = mutable_param
        
    @classmethod
    def get(cls, qualified_name): 
        return cls._params.get(qualified_name, None)

    @classmethod
    def get_all(cls, namespace=None):
        if namespace is not None:
            return {k: v for k, v in cls._params.items() if k.startswith(namespace)}
        return cls._params
    
class OptunaParam(numbers.Real):
    def __init__(self, value_type, name, run_name, experiment_name, categories=None, domain:Tuple=None, step=1, log=False):
        self.value_type = value_type if value_type is not None else domain[0]
        self.name = name
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.categories = categories
        self.domain = domain
        self.step = step
        self.log = log
        OptunaState.set(f"{self.experiment_name}_{self.run_name}_{self.name}", self)

    @property
    def __class__(self):
        return type(self.value)
    
    def set(self, value):
        self.value = value

    def __getattr__(self, name):
        return getattr(self.value, name)
    
    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __truediv__(self, other):
        return float(self) / other

    def __rtruediv__(self, other):
        return other / float(self)

    def __add__(self, other):
        return float(self) + other

    def __radd__(self, other):
        return other + float(self)

    def __sub__(self, other):
        return float(self) - other

    def __rsub__(self, other):
        return other - float(self)

    def __mul__(self, other):
        return float(self) * other

    def __rmul__(self, other):
        return other * float(self)

    def __eq__(self, other):
        return float(self) == other

    def __lt__(self, other):
        return float(self) < other

    def __bool__(self):
        return bool(self.value)

    # Additional methods required by numbers.Real abstract base class
    def __abs__(self):
        return abs(float(self))

    def __pos__(self):
        return +float(self)

    def __neg__(self):
        return -float(self)

    def __le__(self, other):
        return float(self) <= other

    def __gt__(self, other):
        return float(self) > other

    def __ge__(self, other):
        return float(self) >= other