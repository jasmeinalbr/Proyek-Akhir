"""
Local pipeline definition and execution
"""

import os
from typing import Text

from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

# Pipeline Config
PIPELINE_NAME = "jasmeinalbaar-pipeline"

# Pipeline inputs
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/transform.py"
TUNER_MODULE_FILE = "modules/tuner.py"
TRAINER_MODULE_FILE = "modules/trainer.py"

# Pipeline outputs
OUTPUT_BASE = "output"
SERVING_MODEL_DIR = os.path.join(OUTPUT_BASE, "serving_model")
PIPELINE_ROOT = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")

# Pipeline Initialization
def init_local_pipeline(
    components, pipeline_root: Text
) -> pipeline.Pipeline:
    logging.info(f"Pipeline root set to: {pipeline_root}")

    beam_args = [
        "--direct_running_mode=multi_processing",
        # 0 = auto-detect based on available CPU
        "--direct_num_workers=0",
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
        beam_pipeline_args=beam_args,
    )

# Run Pipeline
if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    from modules.components import init_components

    components = init_components(
        DATA_ROOT,
        training_module=TRAINER_MODULE_FILE,
        tuner_module=TUNER_MODULE_FILE,
        transform_module=TRANSFORM_MODULE_FILE,
        training_steps=5000,
        eval_steps=1000,
        serving_model_dir=SERVING_MODEL_DIR,
    )

    p = init_local_pipeline(components, PIPELINE_ROOT)
    BeamDagRunner().run(pipeline=p)