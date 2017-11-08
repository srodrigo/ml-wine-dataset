#!/bin/bash

mkdir -p graphs

description="Enter the script to run:
1) Data analysis
2) Experiment 1 with raw data
3) Experiment 1 with raw data and cross-validation
4) Experiment 2 with feature selection
5) Experiment 2 with feature selection and cross-validation
6) Experiment 3 with parameter tuning
"
read -p "$description" script_number

case $script_number in
    1)
        script_name='data-analysis.py'
        ;;
    2)
        script_name='exp1_raw-data.py'
        ;;
    3)
        script_name='exp2_feature-selection.py'
        ;;
    4)
        script_name='exp2_feature-selection_cv.py'
        ;;
    5)
        script_name='exp3_param-tuning.py'
        ;;
esac

docker build -t wine-dataset .
docker run \
  -v ${PWD}/src:/app/src:ro \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/graphs:/app/graphs \
  wine-dataset \
  python3 "src/main/${script_name}" ./data/wine.data ./graphs/

