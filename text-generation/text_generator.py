#####################################
#               TO-DO               #
# Change code to work for GPU as    #
# well as CPU. I am not sure, how   #
# to do that. The next three        #
# comments shows the resources I    #
# used for createing this combined  #
# optimized code.                   #
#####################################
#               USAGE               #
# To train the code:                #
# $ python3 text_generator.py -t    #
#    True -tf wonderland.txt        #
#                                   #
# To run it with weight file:       #
# $ python3 text_generator.py       #
#                                   #
# Please run this command for more  #
# details:                          #
# $ python3 text_generator.py -h    #
#                                   #
#                           - Mert  #
#####################################

# Resources:
# 1. https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
# 2. https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
# 3. https://medium.com/@izzygrandic/ai-writing-stories-based-on-pictures-52b3ddbcd7d

# Larger LSTM Network to Generate Text for Alice in Wonderland
import os
import sys
import numpy
import argparse
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#####################################
#               TO-DO               #
# Data preprocessing. This code is  #
# not doing any. It may perform bad #
# just because of this. Also, we    #
# may look for other stories maybe? #
# It is not a must tho.             #
#                           - Mert  #
#####################################

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train",
                    type=int,
                    nargs='?',
                    default=0,
                    help="Initialize training.")
parser.add_argument("-tf", "--train_file",
                    type=str,
                    nargs='?',
                    default="wonderland.txt",
                    help="The text file on which the model will be trained.")
parser.add_argument("-cp", "--caption_predictions",
                    type=str,
                    nargs='?',
                    default="prediction0.txt",
                    help="The text file which contains the caption predictions.")
parser.add_argument("-w", "--weight",
                    type=str,
                    nargs='?',
                    default="weights-improvement-55-0.3146-bigger.hdf5",
                    help="The weight file for the pre-trained model.")
args = parser.parse_args()
print(args)

"""
Make Tensorflow less verbose
"""
try:
    # noinspection PyPackageRequirements
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
    # noinspection PyUnusedLocal
    def deprecated(date, instructions, warn_once=True):
        def deprecated_wrapper(func):
            return func
        return deprecated_wrapper

    from tensorflow.python.util import deprecation
    deprecation.deprecated = deprecated
except ImportError:
    pass

# load ascii text and covert to lowercase
filename = args.train_file
raw_text = ""
with open(filename, 'r', encoding='utf-8') as f:
    raw_text = f.read().lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: {}".format(n_chars))
print("Total Vocab: {}".format(n_vocab))

# Making captions one line and configuring the sequence length
seq_length   = 100
captions     = ""
captions_len = 0
with open(args.caption_predictions,"r") as file:
    captions = " ".join([line.replace("\n", "").replace(" .", ".") for line in file.readlines()])
    captions_len = len(captions)
    seq_length   = captions_len
    print("Captions have been read. Length: {}".format(seq_length))

# prepare the dataset of input to output pairs encoded as integers
dataX = []
dataY = []
for i in tqdm(range(0, n_chars - seq_length, 1), desc="Pattern Checking"):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: {}".format(n_patterns))
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

#####################################
#               TO-DO               #
# We may need to tune the model     #
# I am not sure if it will perform  #
# well enough.                      #
#                           - Mert  #
#####################################
# define the LSTM model
model = Sequential()
model.add(LSTM(700, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(700))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

if (args.train):
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # define the checkpoint
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit the model
    model.fit(X, y, epochs=75, batch_size=256, callbacks=callbacks_list)
else:
    # load the network weights
    model.load_weights(args.weight)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    #####################################
    # FIRST APPROACH
    # # pick a random seed
    # with open("prediction0.txt","r") as file:
    #         with open("results.txt","w") as resultFile:
    #                 count = 1
    #                 for line in file:
    #                      resultFile.write("{}: {}\n".format(count,line))
    #                      count += 1
    #                      print(count)
    #                      pattern = [char_to_int[char] for char in line]
    #                      end = numpy.random.randint(0,len(chars),100-len(pattern))
    #                      pattern += [char_to_int[chars[e]] for e in end]
    #                      for i in range(1000):
    #                             x = numpy.reshape(pattern, (1, len(pattern), 1))
    #                             x = x / float(n_vocab)
    #                             prediction = model.predict(x, verbose=0)
    #                             index = numpy.argmax(prediction)
    #                             result = int_to_char[index]
    #                             seq_in = [int_to_char[value] for value in pattern]
    #                             resultFile.write(result)
    #                             #sys.stdout.write(result)
    #                             pattern.append(index)
    #                             pattern = pattern[1:len(pattern)]
    #                      resultFile.write("\n")
    #####################################

    #####################################
    # SECOND APPROACH
    with open("results.txt","w") as resultFile:
        resultFile.write("Seed (Length = {}):\n \"{}\"\n".format(captions_len, captions))
        resultFile.flush()
        os.fsync(resultFile)
        pattern = [char_to_int[char] for char in captions]
        #####################################
        #               TO-DO               #
        # This 1000 determines how many     #
        # characters will be given as       #
        # output. It is said to be better   #
        # if we make it smaller. We have to #
        # test it out tho.                  #
        #                           - Mert  #
        #####################################
        for i in tqdm(range(1000), desc="Generating"):
            x = numpy.reshape(pattern, (1, len(pattern), 1))
            x = x / float(n_vocab)
            prediction = model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            result = int_to_char[index]
            seq_in = [int_to_char[value] for value in pattern]
            resultFile.write(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        resultFile.write("\n")
    print("\nDone.")
