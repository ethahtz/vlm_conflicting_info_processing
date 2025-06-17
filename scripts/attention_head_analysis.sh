#!/bin/bash


# IMPORTANT
seed=42
work_dir=""
# model_name_list=("llava_1.5_7b")  # ("llava_1.5_7b" "instrucblip_vicuna_7b" "qwen2_5" "llava_ov_7b")
# model_name_list=("llava_1.5_7b" "instrucblip_vicuna_7b")
# model_name_list=("instrucblip_vicuna_7b")
# model_name_list=("llava_1.5_7b")
model_name_list=("llava_ov_7b")
# model_name_list=("qwen2_5")
# dataset="Pascal"  # ("cifar10" "cifar100" "imagenet100" "Pascal" "CUB_color" "lm_CelebA_Lipstick")
dataset="cifar100"
version=0
precompute_version=0
num_samples=100

# !! REMEMBER TO MODIFY LAYER AND HEAD BELOW !!
alpha_lower_bound=-10
alpha_upper_bound=10
alpha_interval=1

# Prompt-related arguments
caption_type="inconsistent"
order="icq"
n_candidates=5
is_assistant_prompt="1"

candidates_template="Select from the following classes: "
further_instruction="Answer the question using a single word or phrase."
answer_template="Answer:"
use_pointers=1

# Evaluation-related arguments
evaluation="open_ended"  # ("open_ended" "multiple_choice")
use_log_prob_sum=1
batch_size=1


python src/head_attribution_intervention.py \
    --seed ${seed} \
    --work_dir ${work_dir} \
    --model_name ${model_name} \
    --dataset ${dataset} \
    --version ${version} \
    --precompute_version ${precompute_version} \
    --num_samples ${num_samples} \
    --layer_id ${layer_id} \
    --head_id ${head_id} \
    --alpha_lower_bound ${alpha_lower_bound} \
    --alpha_upper_bound ${alpha_upper_bound} \
    --alpha_interval ${alpha_interval} \
    --caption_type ${caption_type} \
    --order ${order} \
    --n_candidates ${n_candidates} \
    --is_assistant_prompt ${is_assistant_prompt} \
    --candidates_template "${candidates_template}" \
    --further_instruction "${further_instruction}" \
    --answer_template ${answer_template} \
    --use_pointers ${use_pointers} \
    --evaluation ${evaluation} \
    --use_log_prob_sum ${use_log_prob_sum} \
    --batch_size ${batch_size}