# cs5980-term-project
A Picture is Worth a Thousand Words: Gathering Stories from Images through Deep Learning

## Requirements
[im2txt Library](https://github.com/tensorflow/models/tree/master/research/im2txt)

- Bazel ([instructions](http://bazel.io/docs/install.html))
- Python 2.7 or Python 3.X
- TensorFlow 1.0 or greater ([instructions](https://www.tensorflow.org/install/))
- NumPy ([instructions](http://www.scipy.org/install.html))
- Natural Language Toolkit (NLTK):
    - First install NLTK ([instructions](http://www.nltk.org/install.html))
    - Then install the NLTK data package "punkt" ([instructions](http://www.nltk.org/data.html))
- Unzip

### Tensorflow
Install the Tensorflow 1.15, new API crashes with the implementation.

```bash
sudo -H python3 -m pip uninstall tensorflow protobuf && sudo -H python3 -m pip install tensorflow==1.15 protobuf
```

### Checkpoints
Before starting, please download checkpoints from [here](https://merterdemir.com/sharing/checkpoints.zip). Then, open a new directory named as `checkpoints` and copy all the files into that directory or unzip the zip file.

### Pretrained-Models

- [https://github.com/tensorflow/models/issues/466](https://github.com/tensorflow/models/issues/466)
- [https://github.com/tensorflow/models/issues/6513](https://github.com/tensorflow/models/issues/6513)

### Fixing Checkpoints
`fix_checkpoints.py` is used if there is a problem related to checkpoints.

```bash
python3 fix_checkpoints.py
```
- [https://stackoverflow.com/questions/45864363/tensorflow-how-to-convert-meta-data-and-index-model-files-into-one-graph-pb](https://stackoverflow.com/questions/45864363/tensorflow-how-to-convert-meta-data-and-index-model-files-into-one-graph-pb)
- [https://stackoverflow.com/questions/44735794/regarding-using-the-pre-trained-im2txt-model](https://stackoverflow.com/questions/44735794/regarding-using-the-pre-trained-im2txt-model)

## Usage
**Version update:** Now the script does multiprocessing, data preprocess and cleaning. Only running the `run.sh` will be enough. Also, beam size is set to 5.

Put the images you want to generate captions in the `images` folder, and then:

```bash
./run.sh
```

An example of the output should be something like this:

```text
Extracting Bazel installation...
Starting local Bazel server and connecting to it...
INFO: Analyzed target //im2txt:run_inference (18 packages loaded, 101 targets configured).
INFO: Found 1 target...
Target //im2txt:run_inference up-to-date:
  bazel-bin/im2txt/run_inference
INFO: Elapsed time: 10.891s, Critical Path: 0.13s
INFO: 0 processes.
INFO: Build completed successfully, 5 total actions

Building parser...
g++ parser.cpp -o parser -std=c++14 -pedantic-errors -Wall -Wextra -Werror -O3
Done! (Took 12 seconds for building)

Number of Images: 8
Initilization completed! (8/8)

Parsing started for COCO_val2014_000000000428.
Parsing completed for COCO_val2014_000000000428.
Parsed data is being exported for COCO_val2014_000000000428.
Parsed data is exported for COCO_val2014_000000000428.
Parsing started for COCO_val2014_000000000764.
Parsing completed for COCO_val2014_000000000764.
Parsed data is being exported for COCO_val2014_000000000764.
Parsed data is exported for COCO_val2014_000000000764.
Parsing started for COCO_val2014_000000000074.
Parsing completed for COCO_val2014_000000000074.
Parsed data is being exported for COCO_val2014_000000000074.
Parsed data is exported for COCO_val2014_000000000074.
Parsing started for COCO_val2014_000000224477.
Parsing completed for COCO_val2014_000000224477.
Parsed data is being exported for COCO_val2014_000000224477.
Parsed data is exported for COCO_val2014_000000224477.
Parsing started for COCO_val2014_000000000257.
Parsing completed for COCO_val2014_000000000257.
Parsed data is being exported for COCO_val2014_000000000257.
Parsed data is exported for COCO_val2014_000000000257.
Parsing started for COCO_val2014_000000000488.
Parsing completed for COCO_val2014_000000000488.
Parsed data is being exported for COCO_val2014_000000000488.
Parsed data is exported for COCO_val2014_000000000488.
Parsing started for COCO_val2014_000000000395.
Parsing completed for COCO_val2014_000000000395.
Parsed data is being exported for COCO_val2014_000000000395.
Parsed data is exported for COCO_val2014_000000000395.
Parsing started for COCO_val2014_000000000693.
Parsing completed for COCO_val2014_000000000693.
Parsed data is being exported for COCO_val2014_000000000693.
Parsed data is exported for COCO_val2014_000000000693.

Done! (Took 14 seconds for captioning and parsing)
Cleaning the Bazel build...
INFO: Starting clean.
Cleaning the parser build...
Cleaned!
```
Output file should look like:

| id | "prediction0" | "logprob0" | "prediction1" | "logprob1" | "prediction2" | "logprob2" | "prediction3" | "logprob3" | "prediction4" | "logprob4" |
|----|---------------|------------|---------------|------------|---------------|------------|--------------|------------|---------------|------------|
COCO\_val2014\_000000000074 | "a cat sitting on the ground next to a bike" | "0.000166" | "a dog sitting on a sidewalk next to a bike" | "0.000124" | "a cat sitting on the ground next to a bicycle" | "0.000089" | "a cat sitting on a bench next to a bike" | "0.000077" | "a dog sitting on a bench next to a bike" | "0.000070" |
COCO\_val2014\_000000224477 | "a man riding a wave on top of a surfboard" | "0.034281" | "a person riding a surf board on a wave" | "0.018641" | "a man riding a surfboard on top of a wave" | "0.005880" | "a man on a surfboard riding a wave" | "0.005588" | "a man riding a wave on a surfboard in the ocean" | "0.004941" |
COCO\_val2014\_000000000257 | "a group of people standing around a food truck" | "0.003517" | "a group of people standing outside of a food truck" | "0.001867" | "a group of people standing around a truck" | "0.001019" | "a group of people standing in front of a bus" | "0.000639" | "a group of people standing in front of a food truck" | "0.000566" |
COCO\_val2014\_000000000488 | "a batter  |  catcher and umpire during a baseball game" | "0.005729" | "a baseball player swinging a bat at a ball" | "0.003890" | "a baseball player swinging a bat on a field" | "0.002409" | "a baseball player holding a bat on a field" | "0.002359" | "a baseball player swinging a bat at a ball" | "0.002292" |
COCO\_val2014\_000000000395 | "a man is talking on a cell phone" | "0.001166" | "a man talking on a cell phone on a city street" | "0.001061" | "a man talking on a cell phone in a city" | "0.000977" | "a man talking on a cell phone on a street" | "0.000631" | "a man is talking on his cell phone" | "0.000543" |
COCO\_val2014\_000000000693 | "a young boy is sitting on a skateboard" | "0.000152" | "a young boy is sitting on a skateboard" | "0.000052" | "a little girl is sitting on a suitcase" | "0.000021" | "a little boy sitting on a skateboard in a room" | "0.000015" | "a little girl is sitting on the floor with a suitcase" | "0.000007" |
