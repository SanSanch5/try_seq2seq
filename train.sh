#!/bin/bash

source ./setvars.sh

python -m bin.train \
  --config_paths="
      $dir/seq2seq/example_configs/nmt_small.yml,
      $dir/seq2seq/example_configs/train_seq2seq.yml,
      $dir/seq2seq/example_configs/text_metrics_bpe.yml" \
   --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR
