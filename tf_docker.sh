#!/bin/bash
docker run --gpus all -it --rm --network host -v $PWD:/tf/data -w /tf/data tensorflow/tensorflow:latest-gpu-py3-jupyter

