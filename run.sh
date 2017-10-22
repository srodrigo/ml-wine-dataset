#!/bin/bash

docker build -t wine-dataset .
docker run \
  -v ${PWD}/graphs:/graphs \
  -it wine-dataset

