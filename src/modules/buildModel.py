import feather
import tensorflow as tf
from keras import activations, initializers, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (LSTM, Activation, Bidirectional, Concatenate, Dense,
                          Dropout, Embedding, Flatten, Input, Layer, Reshape)
from keras.losses import binary_crossentropy, mse
from keras.models import Model, Sequential, load_model
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from modules.textGenerator import *

tf.compat.v1.set_random_seed(42)
tokenizer = Tokenizer()

BATCH_SIZE = 128


# Make a corpus
corpus = [w for w in re.sub('\[,.*?“”…//-\]', '', krisko_clean).split('\n') if w.strip() != '' or w == '\n']

tokenizer.fit_on_texts(corpus)
TOTAL_WORDS = len(tokenizer.word_index)+1


# Create the input sequences
input_seq = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_seq.append(n_gram_seq)


MAX_SEQ_LEN = max([len(x) for x in input_seq])

input_sequences = np.array(pad_sequences(input_seq, maxlen=MAX_SEQ_LEN, padding='pre'))

xs = input_sequences[:,:-1]
labels = input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=TOTAL_WORDS)


# Build a model
model = Sequential()
model.add(Embedding(TOTAL_WORDS, 256, input_length=MAX_SEQ_LEN-1))
model.add(Dropout(.5))
model.add(Bidirectional(LSTM(256,return_sequences=True,kernel_initializer='random_uniform')))
model.add(Dropout(.5))
model.add(Bidirectional(LSTM(256,return_sequences=True,kernel_initializer='random_uniform')))
model.add(Dropout(.5))
model.add(Bidirectional(LSTM(256,return_sequences=False,kernel_initializer='random_uniform')))
model.add(Dropout(.5))
model.add(Dense(TOTAL_WORDS, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(lr=0.1,decay=.0001),
    metrics=['accuracy'])

model.fit(xs,ys,epochs=500,verbose=1)

model.save('/resources/models/kriskoModel.h5')

model = load_model('/resources/models/kriskoModel.h5')

# Generate a sentence
textGenerator('Love',100)
textGenerator('искам',MAX_SEQ_LEN-1)
textGenerator('искаш',MAX_SEQ_LEN-1)
textGenerator('Аз съм',MAX_SEQ_LEN-2)
textGenerator('Ти си',MAX_SEQ_LEN-2)
