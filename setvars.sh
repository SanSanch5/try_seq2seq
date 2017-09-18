dir=$(realpath `dirname $0`)

export VOCAB_SOURCE=${dir}/parsed/vocab.source.txt
export VOCAB_TARGET=${dir}/parsed/vocab.target.txt
export TRAIN_SOURCES=${dir}/parsed/train.data.txt
export TRAIN_TARGETS=${dir}/parsed/train.labels.txt
export DEV_SOURCES=${dir}/parsed/test.data.txt
export DEV_TARGETS=${dir}/parsed/test.labels.txt

export DEV_TARGETS=${dir}/parsed/test.labels.txt
export TRAIN_STEPS=1000

export MODEL_DIR=${TMPDIR:-/tmp}/try_seq2seq
mkdir -p $MODEL_DIR
