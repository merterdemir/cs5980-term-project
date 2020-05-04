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
# $ python3 text_generator.py train #
#                                   #
# To run it with weight file:       #
# $ python3 text_generator.py       #
#                           - Mert  #
#####################################

# Resources:
# 1. https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
# 2. https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
# 3. https://medium.com/@izzygrandic/ai-writing-stories-based-on-pictures-52b3ddbcd7d

# Larger LSTM Network to Generate Text for Alice in Wonderland
import sys
import numpy
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

# load ascii text and covert to lowercase
filename = "wonderland.txt"
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
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
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

if (len(sys.argv) > 1 and sys.argv[1] == "train"):
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # define the checkpoint
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit the model
    model.fit(X, y, epochs=75, batch_size=64, callbacks=callbacks_list)
else:
    #####################################
    #               TO-DO               #
    # This filename should be the one   #
    # with the least loss value (value) #
    # before the -bigger.hdf5.          #
    #                           - Mert  #
    #####################################
    # load the network weights
    filename = "weights-improvement-55-0.3146-bigger.hdf5"
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    #####################################
    #               TO-DO               #
    # This pattern should contain the   #
    # caption we generated from the     #
    # previous step.                    #
    #                           - Mert  #
    #####################################
    # pick a random seed
    with open("prediction0.txt","r") as file:
            with open("results.txt","w") as resultFile:
                    count = 1
                    for line in file:
                         resultFile.write("{}: {}\n".format(count,line))
                         count += 1
                         print(count)
                         pattern = [char_to_int[char] for char in line]
                         end = numpy.random.randint(0,len(chars),100-len(pattern))
                         pattern += [char_to_int[chars[e]] for e in end]
                         for i in range(1000):
                                x = numpy.reshape(pattern, (1, len(pattern), 1))
                                x = x / float(n_vocab)
                                prediction = model.predict(x, verbose=0)
                                index = numpy.argmax(prediction)
                                result = int_to_char[index]
                                seq_in = [int_to_char[value] for value in pattern]
                                resultFile.write(result)
                                #sys.stdout.write(result)
                                pattern.append(index)
                                pattern = pattern[1:len(pattern)]
                         resultFile.write("\n")
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print("Seed:")
    print("\" {} \"".format(''.join([int_to_char[value] for value in pattern])))

    #####################################
    #               TO-DO               #
    # This 1000 determines how many     #
    # characters will be given as       #
    # output. It is said to be better   #
    # if we make it smaller. We have to #
    # test it out tho.                  #
    #                           - Mert  #
    #####################################
    # generate characters
    
    print("\nDone.")
