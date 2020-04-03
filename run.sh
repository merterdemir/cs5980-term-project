#!/bin/bash

IMAGE_FILES=(`ls ${PWD}/images/*.jpg`)
IMAGE_COUNT=${#IMAGE_FILES[@]}
OUTPUT_FILE="${PWD}/results.csv"
MODE="w"

ST=`date +%s`
./build_inference.sh

echo ""
echo "Building parser..."
cd parser
rm -rf $OUTPUT_FILE
make
cd ~-

ET=`date +%s`
echo "Done! (Took $((ET-ST)) seconds for building)"
echo ""

echo "Number of Images: ${IMAGE_COUNT}"
ST=`date +%s`
for (( i=0; i < $IMAGE_COUNT; i++))
do
    echo -ne "Initializing process: $((i+1))/${IMAGE_COUNT}"\\r
    if [ $i -eq $((IMAGE_COUNT-1)) ]
    then
        echo "Initilization completed! ($((i+1))/${IMAGE_COUNT})"
        echo ""
    fi
    ID=`basename ${IMAGE_FILES[i]%.*}`
    ./parser/parser "${ID}" "`./process_image.sh ${IMAGE_FILES[i]}`" "${OUTPUT_FILE}" "${MODE}" &
    MODE="a"
done

wait
ET=`date +%s`
echo ""
echo "Done! (Took $((ET-ST)) seconds for captioning and parsing)"

./clean.sh
