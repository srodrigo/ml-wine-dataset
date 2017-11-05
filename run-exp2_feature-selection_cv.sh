#!/bin/bash

docker build -t wine-dataset .
docker run \
  -v ${PWD}/src:/app/src:ro \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/graphs:/app/graphs \
  wine-dataset \
  python3 src/main/exp2_feature-selection_cv.py ./data/wine.data ./graphs/

