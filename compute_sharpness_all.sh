#!/bin/bash

declare -a datasets=("FashionMNIST" "CIFAR10")
declare -a modelArch=("SimpleBatch" "MiddleBatch" "ComplexBatch")
declare -a optimizers=("SGD" "PHB" "AdaShift" "AdaBound" "Adagrad" "Adam")

# shellcheck disable=SC2068
for dataset in ${datasets[@]}; do
  for model in ${modelArch[@]}; do
    for opt in ${optimizers[@]}; do
      echo Computing sharpness for model trained on $dataset, $model, $opt
      python main.py compute_sharpness $dataset $model $opt 0 1
    done;
  done;
done