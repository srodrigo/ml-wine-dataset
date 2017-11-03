#!/bin/bash

docker build -t wine-dataset .
docker run \
  -v ${PWD}/src:/app/src:ro \
  -v ${PWD}/data:/app/data \
  wine-dataset \
  python3 src/main/exp1_raw-data.py ./data/wine.data

