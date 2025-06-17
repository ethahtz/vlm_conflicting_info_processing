#!/bin/bash

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
DATASET="cifar100"

SPLIT="train" # train or test
SEED=42
WORK_DIR=""
CAPTION_TYPE="inconsistent" # "inconsistent" or "consistent" depending on the consistency
TARGET_MODALITY="image" #  image or text
ORDER="icq"
CAPTION_TEMPLATE='"This is an image of a {}. " "This is a photo of a {}. " "An image of a {}. " "A photo of a {}. " "This is a {}. " "A {}. "'
N_CANDIDATES=5
QUESTION_TEMPLATE='"What is the class of the input image?" "What is in the image?"'
CANDIDATES_TEMPLATE="Select from the following classes: "

# if you want to compute the representations with some heads intervened: use these variables, otherwise do not pass them in
ALPHA=10.0
layer_idx=11
INTERVENE_HEAD_IDX=14

python precompute_representations.py \
    --split $SPLIT --seed $SEED --work_dir $WORK_DIR --model_name $MODEL_NAME --dataset $DATASET \
    --caption_type $CAPTION_TYPE --modality_to_report $TARGET_MODALITY --order $ORDER \
    --caption_template $CAPTION_TEMPLATE --n_candidates $N_CANDIDATES \
    --question_template $QUESTION_TEMPLATE \
    --candidates_template $CANDIDATES_TEMPLATE \
    --alpha $ALPHA --layer_idx $INTERVENE_LAYER_IDX --head_idx $INTERVENE_HEAD_IDX \