#!/usr/bin/env bash

python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config > ./train_reference_model.log 2>&1

