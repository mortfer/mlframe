from abc import ABC, abstractmethod
from functools import partial
from contextlib import ExitStack
from mlflow.utils import MLflowHandler
import optuna
from mlflow.training.loaders import ITrainingLoader
from mlflow.training.loops import ITrainingLoop
from mlflow.training.handlers import IOutputHandler
from mlflow.training.utils import OptunaState

class ITrainer(ABC):
    """Interface for the model trainer"""
    def __call__(self, *args, **kwargs):
        self.train(*args, **kwargs)

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

class Trainer(ITrainer):
    def __init__(self, 
                 training_loader: ITrainingLoader,
                 training_loop: ITrainingLoop,
                 output_handlers: IOutputHandler = None
                 ):
        self.training_loader = training_loader
        self.training_loop = training_loop
        self.output_handlers = output_handlers if output_handlers is not None else []

    def train(self, run_name, experiment_name, force_run:bool=False, **experiment):
        mlflow_handler = MLflowHandler(run_name=run_name, experiment_name=experiment_name)
        with mlflow_handler.start_run(force_run=force_run) as run:
            model, train_dataset, validation_dataset = self.training_loader(**experiment)
            
            model, history = self.training_loop(model, train_dataset, validation_dataset, **experiment)
            
            for handler in self.output_handlers:
                handler(model, history, **experiment)
            return model, history

class OptunaTrainer(ITrainer):
    def __init__(self,
                 training_loader: ITrainingLoader,
                 training_loop: ITrainingLoop,
                 scoring,
                 n_trials,
                 training_context_manager=None,
                 trial_context_manager=None,
                 study_direction:str="minimize",
                 output_handlers: IOutputHandler = None
                 ):
        self.training_loader = training_loader
        self.training_loop = training_loop
        self.study_direction = study_direction
        self.scoring = scoring
        self.n_trials = n_trials
        self.training_context_manager = training_context_manager
        self.trial_context_manager = trial_context_manager
        self.output_handlers = output_handlers if output_handlers is not None else []

    def objective(self, trial, run_name, experiment_name, **experiment):
        current_hyperparams_state = OptunaState.get_all(namespace=experiment_name+'_'+run_name)
        for namespace, hyperparam in current_hyperparams_state.items():
            method = getattr(trial, ' suggest_'+hyperparam.value_dtype)
            if hyperparam.value_dtype == "categorical":
                suggested_value = method(hyperparam.name, choices=hyperparam.categories)
            else:
                suggested_value = method(hyperparam.name, low=hyperparam.domain[0], high=hyperparam.domain[1], step=hyperparam.step, log=hyperparam.log)
            hyperparam.value = suggested_value

        if self.training_context_manager is not None:
            trial_name = "_".join([f"{key}_{value}" for key, value in current_hyperparams_state.items()])
            trial_manager = self.training_context_manager(run_name=trial_name, experiment_name=experiment_name, nested=True)
            trial_manager = trial_manager(**experiment)
            child_run = self.stack.enter_context(trial_manager)
        model, training_dataset, validation_dataset = self.training_loader(**experiment)
       
        
        model, history = self.training_loop(model, training_dataset, validation_dataset, **experiment)
        
        for handler in self.output_handlers:
            handler(model, history, **experiment)
        score = self.scoring(model, history)
        return score

    def train(self, **experiment):
        with ExitStack() as stack:
            self.stack = stack
            if self.training_context_manager is not None:
                train_manager = self.training_context_manager(**experiment)
                train_manager = train_manager(**experiment)
                parent_run = self.stack.enter_context(train_manager)
            
            partial_objective = partial(self.objective, **experiment)
            study = optuna.create_study(direction=self.study_direction)
            study.optimize(partial_objective, n_trials=self.n_trials, timeout=600)
            trial = study.best_trial
            print(f"Best value: {trial.value}")
            for key, value in trial.params.items():
                print(f"{key}: {value}")
            return study