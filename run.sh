#!/bin/bash

IMAGE_FILES=$(ls ${PWD}/images/*.jpg)

./build_inference.sh

for image in $IMAGE_FILES
do
    ./process_image.sh $image
    echo ""
done