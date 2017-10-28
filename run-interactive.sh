#!/bin/bash

mkdir -p graphs

docker build -t wine-dataset .
docker run \
  -v ${PWD}/src:/app/src:ro \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/graphs:/app/graphs \
  -it wine-dataset

