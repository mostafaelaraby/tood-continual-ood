#!/bin/bash
CUR_DIR=$(pwd)
PROJ_DIR=$(dirname "$CUR_DIR")
CMD="docker run -it --runtime=nvidia --volume=$PROJ_DIR:/cl_ood cl_ood:latest"
echo "$CMD"
eval "$CMD"