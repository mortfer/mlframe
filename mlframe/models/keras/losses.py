import keras
import keras.backend as K
import tensorflow as tf

class PinballLoss(keras.losses.Loss):
    """Custom Pinball loss that wraps another loss function and applies different weights
    for over and underestimation.
    
    Args:
        alpha: Float between 0 and 1. Weight for underestimation (1-alpha for overestimation)
        reduction: Type of reduction to apply to the loss
    """
    def __init__(self, base_loss:str = 'mae', alpha=0.5, reduction=tf.keras.losses.Reduction.AUTO, name='pinball_loss'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.base_loss = base_loss
        self.available_base_losses = ('mae', 'mse')
        assert self.base_loss in self.available_base_losses

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        y_true = tf.cast(y_true, y_pred.dtype)
        # Compute the pinball loss for each data point
        if self.base_loss == 'mae':
            base_loss_values = tf.math.abs(y_true - y_pred)
        elif self.base_loss == 'mse':
            base_loss_values = tf.math.square(y_true - y_pred)

        # Determine over and underestimation
        is_over = tf.cast(y_true < y_pred, tf.float32)
        is_under = 1.0 - is_over

        # Apply weights based on over/underestimation
        weighted_loss = (self.alpha * is_over + (1.0 - self.alpha) * is_under) * base_loss_values

        return tf.reduce_mean(weighted_loss)

    def get_config(self):
        """Returns the config dictionary for serialization"""
        base_config = super().get_config()
        base_config.update({
            "alpha": self.alpha
        })
        return base_config

@tf.function
def tilted_loss(q, y_true, y_pred):
    error = (y_true - y_pred)
    return K.mean(K.maximum(q * error, (q-1) * error))

@tf.function
def quantile_loss(y_true, y_pred, coverage_param=0.75):
    qs = (1-coverage_param, coverage_param)
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true)
    y_true = tf.cast(y_true, y_pred.dtype)
    loss = 0 
    for i, q in enumerate(qs):
        pred_slice = y_pred[:,i]
        loss += tilted_loss(q, y_true, pred_slice)
    return loss
