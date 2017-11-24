#!/bin/bash

data_dir='data'
graphs_dir='graphs'
results_dir='results'

mkdir -p ./${graphs_dir}
mkdir -p ./${results_dir}

docker build -t wine-dataset .
docker run \
  -v ${PWD}/src:/app/src:ro \
  -v ${PWD}/${data_dir}:/app/${data_dir} \
  -v ${PWD}/${graphs_dir}:/app/${graphs_dir} \
  -v ${PWD}/${results_dir}:/app/${results_dir} \
  -it wine-dataset

