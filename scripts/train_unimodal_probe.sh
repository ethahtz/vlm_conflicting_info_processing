#!/bin/bash


SEED=44
CTRL_TASK=0 # is not control task (i.e. labels randomly shuffled)
N_EPOCHS=1000
DATASET="cifar10"
LAYER_IDX=10
IS_IMAGE_PROBE=1 # 1 if image probe, 0 if caption probe
TRAIN_REPRESENTATION_PATH="" # replace this path with a path to the saved precomputed representation file
TEST_REPRESENTATION_PATH=""

python src/train_precompute_probe.py \
        --path_to_train_data $TRAIN_REPRESENTATION_PATH\
        --path_to_test_data  $TEST_REPRESENTATION_PATH\
        --is_image_probe $IS_IMAGE_PROBE --n_epochs $N_EPOCHS --seed $SEED --repr_layer_idx $LAYER_IDX --control_task $CTRL_TASK