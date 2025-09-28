"""Trainer module for Adult Income Dataset"""

import os
import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner

from transform import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    transformed_name,
)


def get_model(hp=None, show_summary=True):
    """Build and compile Keras model. hp digunakan saat tuning."""

    # Jika hp adalah dict, konversi ke HyperParameters
    if isinstance(hp, dict):
        hp_obj = keras_tuner.HyperParameters()
        hp_obj = hp_obj.from_config(hp)
        hp = hp_obj

    # Hyperparameter dari tuner (fallback ke default)
    learning_rate = 0.001 if hp is None else hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
    units_1 = 128 if hp is None else hp.Int("units_1", min_value=64, max_value=256, step=32)
    units_2 = 64 if hp is None else hp.Int("units_2", min_value=16, max_value=128, step=16)

    # Input features
    input_features = []
    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(tf.keras.Input(shape=(dim + 1,), name=transformed_name(key)))
    for feature in NUMERICAL_FEATURES:
        input_features.append(tf.keras.Input(shape=(1,), name=transformed_name(feature)))

    x = tf.keras.layers.concatenate(input_features)

    # Hidden layers
    x = tf.keras.layers.Dense(units_1, activation="relu")(x)
    x = tf.keras.layers.Dense(units_2, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    if show_summary:
        model.summary()

    return model


def gzip_reader_fn(filenames):
    """Loads compressed data (GZIP)."""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Generate features and labels for training/eval."""
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name("income"),
    )
    return dataset


def get_serve_tf_examples_fn(model, tf_transform_output):
    """Serving function for inference."""
    # Tidak perlu tft_layer karena input sudah transformed dari Evaluator

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")])
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.transformed_feature_spec().copy()
        # Pop label jika ada, karena untuk inferensi
        feature_spec.pop(transformed_name("income"), None)
        
        # Parse dengan transformed_feature_spec
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        # Debugging: Cetak kunci fitur parsed
        tf.print("Parsed features keys:", parsed_features.keys())

        # Susun input model dengan urutan yang sama seperti di get_model
        model_inputs = []
        for key in CATEGORICAL_FEATURES.keys():
            transformed_key = transformed_name(key)
            if transformed_key not in parsed_features:
                raise ValueError(f"Missing transformed feature: {transformed_key}")
            model_inputs.append(parsed_features[transformed_key])
        for feature in NUMERICAL_FEATURES:
            transformed_key = transformed_name(feature)
            if transformed_key not in parsed_features:
                raise ValueError(f"Missing transformed feature: {transformed_key}")
            model_inputs.append(parsed_features[transformed_key])

        # Jalankan inferensi dengan input sebagai list
        outputs = model(model_inputs)
        return {"outputs": outputs}

    return serve_tf_examples_fn


def run_fn(fn_args):
    """Main entrypoint for TFX Trainer."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, 64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, 64)

    # Build model (dengan hyperparameter terbaik dari tuner jika ada)
    model = get_model(hp=fn_args.hyperparameters)

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq="batch")

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
        epochs=10,
    )

    # Serving signature
    signatures = {
        "serving_default": get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        )
    }

    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
