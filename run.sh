#!/bin/bash

mkdir -p graphs

docker build -t wine-dataset .
docker run \
  -v ${PWD}/src:/app/src:ro \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/graphs:/app/graphs \
  wine-dataset \
  python3 src/main/data-analysis.py ./data/wine.data ./graphs/

