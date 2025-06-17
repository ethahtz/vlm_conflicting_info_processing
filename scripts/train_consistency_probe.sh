#!/bin/bash

SEED=44
CTRL_TASK=0 # is not control task (i.e. labels randomly shuffled)
N_EPOCHS=1000
DATASET="cifar10"
LAYER_IDX=10
PATH_TO_TRAIN_INCONG="" # replace this path with a path to the saved precomputed representation file (inconsistent representations, train split)
PATH_TO_TEST_INCONG="" # (inconsistent representations, test split)
PATH_TO_TRAIN_CONG="" # (consistent representations, train split)
PATH_TO_TEST_CONG="" # (consistent representations, test split)
FOLD_IDX=0 # if 3-fold, the fold idx ranges from 0 to 2


python src/probe/train_alignment_probe_leave_one_out.py \
    --path_to_train_data_inconsistent $PATH_TO_TRAIN_INCONG \
    --path_to_test_data_inconsistent $PATH_TO_TEST_INCONG \
    --path_to_train_data_consistent $PATH_TO_TRAIN_CONG \
    --path_to_test_data_consistent $PATH_TO_TEST_CONG \
    --n_epochs $N_EPOCHS --seed $SEED --repr_layer_idx $LAYER_IDX --fold_idx $FOLD_IDX --control_task $CTRL_TASK



# or, to train a probe without n-folding on the class label space:

# python src/probe/train_alignment_probe.py \
#     --path_to_train_data_inconsistent $PATH_TO_TRAIN_INCONG \
#     --path_to_test_data_inconsistent $PATH_TO_TEST_INCONG \
#     --path_to_train_data_consistent $PATH_TO_TRAIN_CONG \
#     --path_to_test_data_consistent $PATH_TO_TEST_CONG \
#     --n_epochs $N_EPOCHS --seed $SEED --repr_layer_idx $LAYER_IDX --control_task $CTRL_TASK