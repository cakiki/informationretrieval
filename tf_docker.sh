#!/bin/bash
docker run -it --rm --network host -v $PWD:/tf/data -w /tf/data tensorflow/tensorflow:latest-py3-jupyter

