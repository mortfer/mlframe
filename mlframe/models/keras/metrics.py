import keras
import tensorflow as tf

class WAPE(keras.metrics.Metric):
    def __init__(self, name='wape', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_actual_value = self.add_weight(name='total_actual_value', initializer='zeros')
        self.total_absolute_error = self.add_weight(name='total_absolute_error', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        y_true = tf.cast(y_true, y_pred.dtype)
        absolute_error = tf.abs(y_true - y_pred)
        self.total_actual_value.assign_add(tf.reduce_sum(y_true))
        self.total_absolute_error.assign_add(tf.reduce_sum(absolute_error))

    def result(self):
        return self.total_absolute_error / self.total_actual_value

    def reset_state(self):
        self.total_absolute_error.assign(0.)
        self.total_actual_value.assign(0.)

class Bias(keras.losses.Loss):
    """Bias (Mean Error) loss function that calculates the average error (y_pred - y_true).
    Unlike MAE, this preserves the sign of the error, allowing differentiation between
    over and under predictions.
    """
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='bias'):
        """Initialize the mean error loss.
        
        Args:
            reduction: Type of reduction to apply to the loss
            name: Name of the loss function
        """
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        """Calculate the mean error for each sample.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Mean error values (can be positive or negative)
        """
        error = y_pred - y_true
        return error

    def get_config(self):
        """Returns the config dictionary for serialization."""
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        """Creates a loss function from its config."""
        return cls(**config)

class RelativeBias(keras.losses.Loss):
    """Relative Bias (Mean Error) loss function that calculates the average error (y_pred - y_true)/y_true.
    """
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='relative_bias', epsilon=1e-7):
        super().__init__(reduction=reduction, name=name)
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        """Calculate the mean error for each sample.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Mean error values (can be positive or negative)
        """
        error = (y_pred - y_true)/(y_true + self.epsilon)
        return error

    def get_config(self):
        """Returns the config dictionary for serialization."""
        base_config = super().get_config()
        return base_config
    @classmethod
    def from_config(cls, config):
        """Creates a loss function from its config."""
        return cls(**config)
    