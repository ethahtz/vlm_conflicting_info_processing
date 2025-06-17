#!/bin/bash

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
SEED=42
DATASET="cifar100"


python cluster.py --model_name $MODEL_NAME --seed $SEED --dataset $DATASET