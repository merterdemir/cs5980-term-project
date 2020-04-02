VOCAB_FILE="${PWD}/checkpoints/5M/word_counts.txt"
CHECKPOINT_PATH="${PWD}/checkpoints/5M/model.ckpt-5000000"

if [ $# -eq 0 ]
then
  echo "path or url to image is missing"
  echo "example: ./process_image.sh imgs/bikes.jpg"
  echo "example: ./process_image.sh https://github.com/tensorflow/models/raw/master/im2txt/g3doc/COCO_val2014_000000224477.jpg"
  exit 3
fi

INPUTFILE=$1
# echo $INPUTFILE | grep '^https\{0,1\}://'

# echo "Processing $INPUTFILE"

# Run inference to generate captions.¬
im2txt/bazel-bin/im2txt/run_inference --checkpoint_path=${CHECKPOINT_PATH} --vocab_file=${VOCAB_FILE} --input_files="$INPUTFILE"

