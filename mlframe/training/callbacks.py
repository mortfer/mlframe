import keras
import math
import keras.backend as K
import warnings
import mlflow
from keras.utils import io_utils, tf_utils

class LRCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})

class MLFlowModelCheckpoint(keras.callbacks.Callback): #keras 2

    def _save_model(self, epoch, batch, logs):
        """Saves the model.

        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
              is set to `epoch`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if (
            isinstance(self.save_freq, int)
            or self.epochs_since_last_save >= self.period
        ):
            # Block only when saving interval is reached.
            logs = tf_utils.sync_to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, batch, logs)

            try:
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        warnings.warning(
                            "Can save best model only with %s available, "
                            "skipping.",
                            self.monitor,
                        )
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                io_utils.print_msg(
                                    f"\nEpoch {epoch + 1}: {self.monitor} "
                                    "improved "
                                    f"from {self.best:.5f} to {current:.5f}, "
                                    f"saving model to {filepath}"
                                )
                            self.best = current

                            # Handles saving and corresponding options
                            mlflow.tensorflow.log_model(self.model, filepath, input_example=getattr(self, ' input_example', None), keras_model_kwargs={'save_format': 'h5'})
                        else:
                            if self.verbose > 0:
                                io_utils.print_msg(
                                    f"\nEpoch {epoch + 1}: "
                                    f"{self.monitor} did not improve "
                                    f"from {self.best:.5f}"
                                )
                else:
                    if self.verbose > 0:
                        io_utils.print_msg(
                            f"\nEpoch {epoch + 1}: saving model to {filepath}"
                        )

                    # Handles saving and corresponding options
                    mlflow.tensorflow.log_model(self.model, filepath, input_example=getattr(self, ' input_example', None), keras_model_kwargs={'save_format': 'h5'})

                
            except IsADirectoryError:  # h5py 3.x
                raise IOError(
                    "Please specify a non-directory filepath for "
                    "ModelCheckpoint. Filepath used is an existing "
                    f"directory: {filepath}"
                )
            except IOError as e:  # h5py 2.x
                # `e.errno` appears to be `None` so checking the content of
                # `e.args[0]`.
                if "is a directory" in str(e.args[0]).lower():
                    raise IOError(
                        "Please specify a non-directory filepath for "
                        "ModelCheckpoint. Filepath used is an existing "
                        f"directory: f{filepath}"
                    )
                # Re-throw the error for any other causes.
                raise e
    @property
    def input_example(self):
        return self._input_example
    @input_example.setter
    def input_example(self, input_example):
        self._input_example = input_example
        
class MLFlowCallback(keras.callbacks.Callback):
    def __init__(self, log_every_epoch=True, log_every_n_steps=None):
        super().__init__()
        self.log_every_epoch = log_every_epoch
        self.log_every_n_steps = log_every_n_steps
        if log_every_epoch and log_every_n_steps is not None:
            raise ValueError(
                "'log_every_epoch' and 'log_every_n_steps' must be None if 'log_every_epoch=True', received "
                f"'log_every_n_steps={log_every_n_steps}'"
            )
        if not log_every_epoch and log_every_n_steps is None:
            raise ValueError(
                "'log_every_epoch' and 'log_every_n_steps' must be specified if 'log_every_epoch=False', received "
                "'log_every_n_steps=None'"
            )
        

    def on_train_begin(self, logs=None):
        """Log model optimizer configuration when training begins."""
        config = self.model.optimizer.get_config()
        mlflow.log_params({f"optimizer_{k}": v for k, v in config.items()})

        model_summary = []
        
        def print_fn(line, *args, **kwargs):
            model_summary.append(line)
        self.model_summary(print_fn=print_fn)
        summary = "\n".join(model_summary)
        mlflow.log_text(summary, artifact_file="model_summary.txt")
        fig = keras.utils.plot_model(self.model, show_shapes=True, expand_nested=True, dpi=300)
        mlflow.log_figure(fig, artifact_file="model_plot.png")
        mlflow.log_param("loss_name", self.model.loss.name)

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch."""
        if not self.log_every_epoch or logs is None:
            return
        mlflow.log_metrics(logs, step=epoch)

    def on_test_end(self, batch, logs=None):
        """Log validation metrics at validation end."""
        if logs is None:
            return
        metrics = {"validation_" + k: v for k, v in logs.items()}
        mlflow.log_metrics(metrics)
