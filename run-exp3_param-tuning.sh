#!/bin/bash

docker build -t wine-dataset .
docker run \
  -v ${PWD}/src:/app/src:ro \
  -v ${PWD}/data:/app/data \
  wine-dataset \
  python3 src/main/exp3_param-tuning.py ./data/wine.data ./graphs/

