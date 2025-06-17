#!/bin/bash

SEED=42
WORK_DIR=""
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
DATASET="cifar100"

VERSION=0

CAPTION_TYPE="inconsistent" # "inconsistent" "consistent" "text_only" "no_caption"
MODALITY_TO_REPORT="image" #  image or text
ORDER="icq"
CAPTION_TEMPLATE='"This is an image of a {}. " "This is a photo of a {}. " "An image of a {}. " "A photo of a {}. " "This is a {}. " "A {}. "'
N_CANDIDATES=5
QUESTION_TEMPLATE='"What is the class of the input image?" "What is in the image?"'
CANDIDATES_TEMPLATE="Select from the following classes: "
EXPLICIT_HELPER_TEMPLATE=""
IS_EXPLICIT_HELPER=0
IS_ASSISTANT_PROMPT=1
FURTHER_INSTRUCTION="Answer the question using a single word or phrase."
ANSWER_TEMPLATE="Answer:"
EVALUATION="open_ended"
BATCH_SIZE=10
USE_LOG_PROB_SUM=0

# if you want to run evaluation with some heads intervened: use these variables, otherwise do not pass them in
ALPHA=10.0
layer_idx=11
INTERVENE_HEAD_IDX=14

python src/prompt_evaluation.py \
    --seed $SEED \
    --work_dir $WORK_DIR \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --version $VERSION \
    --caption_type $CAPTION_TYPE \
    --modality_to_report $MODALITY_TO_REPORT \
    --order $ORDER \
    --is_explicit_helper $IS_EXPLICIT_HELPER \
    --n_candidates $N_CANDIDATES \
    --is_assistant_prompt $IS_ASSISTANT_PROMPT \
    --caption_template $CAPTION_TEMPLATE \
    --explicit_helper_template $EXPLICIT_HELPER_TEMPLATE \
    --question_template $QUESTION_TEMPLATE \
    --candidates_template $CANDIDATES_TEMPLATE \
    --further_instruction $FURTHER_INSTRUCTION \
    --answer_template $ANSWER_TEMPLATE \
    --evaluation $EVALUATION \
    --batch_size $BATCH_SIZE \
    --use_log_prob_sum $USE_LOG_PROB_SUM \
    --alpha $ALPHA --layer_idx $LAYER_IDX --head_idx $HEAD_IDX
