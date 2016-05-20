# -*- coding: utf-8 -*-
import gzip
import time
from pprint import pprint

from keras.engine import Merge
from keras.layers import Embedding, Flatten, TimeDistributed
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.layers.core import Dropout, Dense, Permute, Reshape
from keras.layers.recurrent import *
from keras.models import *
from keras.utils import np_utils
from sklearn.metrics import f1_score, precision_recall_fscore_support

import PortEvalReader
from KerasLayer.FixedEmbedding import FixedEmbedding

trainFile = 'corpus/corpus_First_HAREM.txt'
#trainFile = 'corpus/corpus_paramopama+second_harem.txt'
testFile = 'corpus/corpus_miniHAREM.txt'

# Word Embeddings
print "Reading word embeddings"
vocabPath =  'embeddings/Portuguese.vocab.gz'

word2Idx = {} #Maps a word to the index in the embeddings matrix
embeddings = [] #Embeddings matrix

#max_charlen = 36

with gzip.open(vocabPath, 'r') as fIn:
    idx = 0
    charIdx = 1
    for line in fIn:
        split = line.strip().split(' ')
        embeddings.append(np.array([float(num) for num in split[1:]]))
        #if len(split[0]) > max_charlen:
        #    max_charlen = len(split[0])
        word2Idx[split[0].decode('utf-8')] = idx
        idx += 1

embeddings = np.asarray(embeddings, dtype='float32')

#Create a mapping for our labels
label2Idx = {u'O':0}
idx = 1

# Adding remaining labels
for nerClass in [u'PESSOA', u'LOCAL', u'ORGANIZACAO', u'TEMPO', u'VALOR']:
    label2Idx[nerClass] = idx
    idx += 1

#Inverse label mapping
idx2Label = {v: k for k, v in label2Idx.items()}

#Hyper parameters
windowSize = 2 # 3 to the left, 3 to the right
number_of_epochs = 50
minibatch_size = 20
n_in = 2*windowSize+1
n_out = len(label2Idx)
n_filters = 10
char_embedding_size = 20
n_hidden = n_in*(embeddings.shape[1]+n_filters)
#n_hidden = 500
char_window = 5

# Read in data
print "Reading data and create matrices"
train_sentences, max_charlen = PortEvalReader.readFile2(trainFile)
test_sentences, max_charlen2 = PortEvalReader.readFile2(testFile)


# Create numpy arrays
train_x, train_y, train_x_char, char2Idx = PortEvalReader.createNumpyArrayAndCharData(train_sentences, windowSize, word2Idx, label2Idx, max_charlen)
test_x, test_y, test_x_char, temp = PortEvalReader.createNumpyArrayAndCharData(test_sentences, windowSize, word2Idx, label2Idx, max_charlen)

max_charlen_padded = max_charlen + windowSize*2
char_vocab_size = len(char2Idx)+1

print "Word Embeddings shape",embeddings.shape

#Word Embeddings
model_wordemb = Sequential()
model_wordemb.add(Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in,
                         weights=[embeddings]))

#Character Embeddings
model_cnn = Sequential()
model_cnn.add(Embedding(output_dim=char_embedding_size, input_dim=char_vocab_size, input_length=n_in*max_charlen_padded, init='glorot_uniform'))
model_cnn.add(Reshape((n_in, max_charlen_padded, char_embedding_size)))
model_cnn.add(TimeDistributed(Convolution1D(nb_filter=n_filters, filter_length=char_window, border_mode='valid')))
model_cnn.add(TimeDistributed(MaxPooling1D(pool_length=max_charlen)))
model_cnn.add(Reshape((n_in, n_filters)))


# Create the  Network
model = Sequential()
model.add(Merge([model_wordemb, model_cnn], mode='concat'))
model.add(LSTM(output_dim=n_hidden, init='glorot_uniform', activation='tanh',
                batch_input_shape=(None,n_in,embeddings.shape[1]+char_embedding_size),return_sequences=True),)
#model.add(Dropout(0.5))
model.add(LSTM(output_dim=n_hidden, init='glorot_uniform', activation='tanh',batch_input_shape=(None,n_in,n_hidden)),)
model.add(Dense(output_dim=n_out, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

model.load_weights('modelos_treinados/10-LSTM_batch20_comjanela_charEmbeddingsApenas/parciais/modelo_parcial5.h5')
print 'Testing...'

### Testing ###
b = model.predict_classes([test_x, test_x_char],verbose=1)

print '\nPrecision, Recall, F-measure por classe (O, PESSOA, LOCAL, ORGANIZACAO, TEMPO, VALOR): '
print precision_recall_fscore_support(test_y, b)
print '\nPrecision, Recall, F-measure Total:'
print precision_recall_fscore_support(test_y, b, average='macro')