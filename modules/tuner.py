"""Tuner module for Adult Income Dataset"""

import tensorflow_transform as tft
import keras_tuner
from tfx.components.tuner.component import TunerFnResult
from modules.trainer import get_model, input_fn

def tuner_fn(fn_args):
    """Build the Keras Tuner."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    tuner = keras_tuner.RandomSearch(
        hypermodel=get_model,
        objective="val_accuracy",
        max_trials=5,
        directory=fn_args.working_dir,
        project_name="adult_income_tuning",
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": input_fn(fn_args.train_files, tf_transform_output),
            "validation_data": input_fn(fn_args.eval_files, tf_transform_output),
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": 5,
        },
    )
