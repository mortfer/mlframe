import math
import numpy as np
import repartipy
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm.reader import make_batch_reader


class SparkSessionManager:
    _spark = None

    @classmethod
    def set_spark(cls, spark_session):
        cls._spark = spark_session

    @classmethod
    def get_spark(cls):
        if cls._spark is None:
            raise RuntimeError(" SparkSession has not been initialized.")
        return cls._spark


class PetastormIterator:
    def __init__(
        self,
        df,
        batch_size,
        feature_names,
        y_name,
        sample_weights=None,
        shuffle=False,
        epochs=None,
    ):
        self.df = df
        self.batch_size = batch_size
        self.feature_names = feature_names
        self.y_name = y_name
        self.sample_weights = sample_weights
        self.shuffle = shuffle
        self.epochs = epochs
        self.data_buffer = {}

    def __len__(self):
        return math.ceil(len(self.converter) / self.batch_size)

    def __iter__(self):
        return self

    def _producer(
        self,
    ):
        if self.data_buffer == {}:
            self.data_buffer = next(self.reader)._asdict()
        if self.data_buffer[self.y_name].shape[0] < self.batch_size:
            try:
                next_chunk = next(self.reader)._asdict()
                self.data_buffer = {
                    k: np.concatenate((self.data_buffer[k], next_chunk[k]))
                    for k, v in next_chunk.items()
                }
            except StopIteration:
                if self.data_buffer[self.y_name].shape[0] == 0:
                    raise StopIteration
                else:
                    batch = self.data_buffer
                    X = {name: batch[name] for name in self.feature_names}
                    y = batch[self.y_name]
                    if self.sample_weights is not None:
                        w = batch[self.sample_weights]
                        return X, y, w
                    else:
                        return X, y
        batch = {k: v[: self.batch_size] for k, v in self.data_buffer.items()}
        self.data_buffer = {
            k: v[self.batch_size :] for k, v in self.data_buffer.items()
        }
        X = {name: batch[name] for name in self.feature_names}
        y = batch[self.y_name]
        if self.sample_weights is not None:
            w = batch[self.sample_weights]
        self.data_buffer = {}
        if self.sample_weights is not None:
            return X, y, w
        else:
            return X, y

    def __next__(self):
        result = self._producer()
        return result

    def on_epoch_end(self):
        pass

    def __enter__(self):
        # SparkSessionManager.get_spark().conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, '..')
        desired_partitions_size_in_bytes = 2**28
        with repartipy.SamplingSizeEstimator(
            spark=SparkSessionManager.get_spark(), df=self.df, sample_count=10
        ) as se:
            desired_partition_count = se.get_desired_partition_count(
                desired_partition_size_in_bytes=desired_partitions_size_in_bytes
            )
            desired_partition_count = max(2, desired_partition_count)
        partitions = self.df.rdd.getNumPartitions()
        print(f"Proposed number of new partitons: {desired_partition_count}")
        print(f"Current partitions: {partitions}")
        if partitions > desired_partition_count:
            self.df = self.df.repartition(desired_partition_count)
            print("Repartitioned df")
        else:
            print("Did not repartition df")

        self.converter = make_spark_converter(self.df)
        self.reader = make_batch_reader(
            self.converter.cache_dir_url,
            num_epochs=self.epochs,
            shuffle_rows=self.shuffle,
            shuffle_row_groups=self.shuffle,
        )
        self.reader = self.reader.__enter__()
        return iter(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reader.__exit__(exc_type, exc_val, exc_tb)
        self.converter.delete()
        self.data_buffer = {}


class PytorchPetastormIterator:
    def __init__(self, df):
        pass
