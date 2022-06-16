#!/bin/bash

declare -a datasets=("FashionMNIST" "CIFAR10")
declare -a modelArch=("SimpleBatch" "MiddleBatch" "ComplexBatch")
declare -a optimizers=("SGD" "PHB" "AdaShift" "AdaBound" "Adagrad" "Adam")

# shellcheck disable=SC2068
for dataset in ${datasets[@]}; do
  for model in ${modelArch[@]}; do
    for opt in ${optimizers[@]}; do

      echo Training $dataset, $model, $opt, no SAM
      python main.py train $dataset $model $opt 0 0

      echo Training $dataset, $model, $opt, with SAM
      python main.py train $dataset $model $opt 1 0
    done;
  done;
done