from abc import ABC, abstractmethod

class ITrainingLoader(ABC):
    def __call__(self, **kwargs):
        return self.load(**kwargs)

    @abstractmethod
    def load(self, **kwargs):
        pass

class TrainingLoader(ITrainingLoader):
    def __init__(self, model_loader, dataset_loader):
        self.model_loader = model_loader
        self.dataset_loader = dataset_loader

    def load(self, **experiment):
        training_dataset, validation_dataset = self.dataset_loader(**experiment)
        model = self.model_loader(training_dataset, validation_dataset, **experiment)
        return model, training_dataset, validation_dataset


