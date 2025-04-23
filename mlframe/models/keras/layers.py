import keras
import tensorflow as tf
import keras.backend as K
from keras.layers import Multiply

#@keras.saving.register_keras_serializable(package="mylayers")
class MonteCarloDroupout(keras.layers.Dropout):
    def __init__(self, dropout_rate, num_samples=128, **kwargs):
        super().__init__(dropout_rate, **kwargs)
        self.num_samples = num_samples
        self.dropout_layer = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        output = keras.layers.RepeatVector(self.num_samples)(inputs)
        output = self.dropout_layer(inputs, training=True)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "dropout_rate": self.dropout_layer.rate,
            "num_samples": self.num_samples
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["dropout_layer"] = keras.layers.deserialize(config["dropout_layer"])
        return cls(**config)


