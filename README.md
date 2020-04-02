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