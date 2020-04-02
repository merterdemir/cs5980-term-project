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
Put the images you want to generate captions in the `images` folder, and then:

```bash
./run.sh
```

An example of the output should be something like this:

```text
INFO: Analyzed target //im2txt:run_inference (18 packages loaded, 101 targets configured).
INFO: Found 1 target...
Target //im2txt:run_inference up-to-date:
  bazel-bin/im2txt/run_inference
INFO: Elapsed time: 0.534s, Critical Path: 0.02s
INFO: 0 processes.
INFO: Build completed successfully, 5 total actions
/path_to_project/cs5980-term-project
Captions for image COCO_val2014_000000000074.jpg:
  0) a dog sitting on a sidewalk next to a bike . (p=0.000124)
  1) a dog sitting on a sidewalk next to a bicycle . (p=0.000075)
  2) a dog sitting on a bench next to a bike . (p=0.000070)

Captions for image COCO_val2014_000000000257.jpg:
  0) a group of people standing around a food truck . (p=0.003517)
  1) a group of people standing outside of a food truck . (p=0.001867)
  2) a group of people standing next to a truck . (p=0.000519)

Captions for image COCO_val2014_000000000395.jpg:
  0) a man in a hat is talking on a cell phone . (p=0.000636)
  1) a man in a hat is talking on his cell phone . (p=0.000281)
  2) a man in a hat is talking on a cell phone (p=0.000253)

Captions for image COCO_val2014_000000000428.jpg:
  0) a little girl that is standing in front of a cake . (p=0.000224)
  1) a little girl that is sitting in front of a cake . (p=0.000195)
  2) a little girl that is standing in front of a mirror . (p=0.000113)

Captions for image COCO_val2014_000000000488.jpg:
  0) a baseball player swinging a bat at a ball (p=0.003890)
  1) a baseball player swinging a bat on a field . (p=0.002409)
  2) a baseball player holding a bat on a field . (p=0.002359)

Captions for image COCO_val2014_000000000693.jpg:
  0) a little boy sitting on top of a skateboard . (p=0.000220)
  1) a little boy sitting on top of a piece of luggage . (p=0.000141)
  2) a little girl is sitting on a suitcase (p=0.000021)

Captions for image COCO_val2014_000000000764.jpg:
  0) a group of young men playing a game of frisbee . (p=0.013892)
  1) a group of young people playing a game of frisbee . (p=0.007982)
  2) a group of people playing frisbee in a field . (p=0.003358)

Captions for image COCO_val2014_000000224477.jpg:
  0) a man riding a wave on top of a surfboard . (p=0.034281)
  1) a person riding a surf board on a wave (p=0.018641)
  2) a man riding a wave on a surfboard in the ocean . (p=0.004941)
```

To clean the Blaze Build, run the `clean.sh` script:

```bash
./clean.sh
```
```text
Cleaning the build...
INFO: Starting clean.
Cleaned!
/path_to_project/cs5980-term-project
```