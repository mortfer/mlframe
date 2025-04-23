from abc import ABC, abstractmethod

class IModelLoader(ABC):
    @abstractmethod
    def load(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.load(*args, **kwargs)

class MultiLoader(IModelLoader):
    """Three stage builder"""
    def __init__(self, input_builder, backbone_builder, output_builder):
        self.input_builder = input_builder
        self.backbone_builder = backbone_builder
        self.output_builder = output_builder

    def load(self, training_dataset, validation_dataset, **experiment):
        vocabularies = training_dataset.metadata.get("vocabularies", None)
        preprocessed_layer, inputs = self.input_builder(vocabularies=vocabularies)
        backbone = self.backbone_builder(preprocessed_layer)
        output_model = self.output_builder(backbone, inputs)
        return output_model

class MockModelBuilder(IModelLoader):
    """Model built in experiment definition"""
    def __init__(self, model):
        self.model = model

    def load(self, *args, **experiment):
        return self.model
