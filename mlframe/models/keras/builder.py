from keras import layers
import math
import tensorflow as tf
import keras
from keras.models import Model
from mlframe.models.keras.layers import MonteCarloDroupout

def encode_features(inputs, categorical_features_vocabulary, categorical_string_features_vocabulary, embedding_dim:int = 8):
    encoded = []
    embedding_dim = int(embedding_dim)
    print("encode_features on_dim", len(embedding_dim))
    for feature_name, input_layer in inputs.items():
        if feature_name in categorical_features_vocabulary.keys():
            vocab_size = len(categorical_features_vocabulary[feature_name]) + 1  # +1 for missing values
            embedding_size = embedding_dim[feature_name] if embedding_dim else int(math.sqrt(vocab_size))
            print(f"{embedding_size}")
            encoded_feature = layers.Embedding(vocab_size, embedding_size)(input_layer)
        elif feature_name in categorical_string_features_vocabulary.keys():
            vocab_length = len(categorical_string_features_vocabulary[feature_name]) + 1  # +1 for missing values
            embedding_size = embedding_dim[feature_name] if embedding_dim else int(math.sqrt(vocab_length))
            encoded_feature = layers.Embedding(vocab_length, embedding_size)(input_layer)
        else:
            encoded_feature = layers.Lambda(lambda x: x)(input_layer)
        encoded[feature_name] = encoded_feature
    return encoded

def create_model_inputs(feature_types: dict):
    inputs = {}
    for feature_name, feature_shape in feature_types.items():
        inputs[feature_name] = layers.Input(name=feature_name, shape=feature_shape)
    return inputs

def lookup(categorical_features_vocabulary, categorical_string_features_vocabulary):
    lookup_dict = {}
    for feature_name, vocab in categorical_features_vocabulary.items():
        lookup = layers.StringLookup(vocabulary=vocab)
        lookup_dict[feature_name] = lookup

    for feature_name, vocab in categorical_string_features_vocabulary.items():
        lookup = layers.StringLookup(vocabulary=vocab)
        lookup_dict[feature_name] = lookup
    return lookup_dict

def build_input(
    feature_types,
    categorical_integer_features,
    categorical_string_features,
    numerical_features,
    countries_features,
    vocabularies,
    embedding_dim: int = 8
):
    features_to_concat = []
    categorical_integer_features_vocabulary = {k: vocabularies[k] for k in categorical_integer_features}
    categorical_string_features_vocabulary = {k: vocabularies[k] for k in categorical_string_features}

    inputs = create_model_inputs(feature_types)
    print("build_input")
    encoded = encode_features(
        inputs,
        categorical_integer_features_vocabulary,
        categorical_string_features_vocabulary,
        embedding_dim
    )
    features_to_concat.extend(encoded.values())

    # Add numerical features
    if numerical_features:
        numerical_features_layer = [tf.expand_dims(inputs[feature_name], -1) for feature_name in numerical_features]
        features_to_concat.extend(numerical_features_layer)

    # Add countries features if present
    if countries_features:
        countries_features_layer = [tf.expand_dims(inputs[feature_name], -1) for feature_name in countries_features]
        features_to_concat.extend(countries_features_layer)

    x = layers.concatenate(features_to_concat, axis=-1)
    return x, inputs

def build_dense_backbone(x):
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    return x

def build_dense_montecarlo_backbone(x, mc_samples=100):
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = MonteCarloDroupout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = MonteCarloDroupout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = MonteCarloDroupout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x)
    return x

def build_output_model(x, inputs):
    x = layers.Dense(1, activation="relu")(x)
    model = Model(inputs=inputs, outputs=x)
    return model


