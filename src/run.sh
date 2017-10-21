#!/bin/bash

docker build -t wine-dataset .
docker run -it wine-dataset

