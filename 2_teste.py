# -*- coding: utf-8 -*-
import gzip

from keras.layers.core import Dropout, TimeDistributedDense, Dense, Flatten
from keras.layers.recurrent import *
from keras.models import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

from sklearn.metrics import precision_recall_fscore_support, f1_score

import PortEvalReader
from KerasLayer.FixedEmbedding import FixedEmbedding

#HYPER PARAMETERS
windowSize = 3 # 3 to the left, 3 to the right
number_of_epochs = 50
#minibatch_size = 5000
minibatch_size = 200

trainFile = 'corpus/corpus_First_HAREM.txt'
testFile = 'corpus/corpus_miniHAREM.txt'

# Word Embeddings
print "Reading word embeddings"
vocabPath =  'embeddings/Portuguese.vocab.gz'

word2Idx = {} #Maps a word to the index in the embeddings matrix
embeddings = [] #Embeddings matrix

with gzip.open(vocabPath, 'r') as fIn:
    idx = 0
    for line in fIn:
        split = line.strip().split(' ')
        embeddings.append(np.array([float(num) for num in split[1:]]))
        word2Idx[split[0]] = idx
        idx += 1

embeddings = np.asarray(embeddings, dtype='float32')

#Create a mapping for our labels
label2Idx = {'O':0}
idx = 1

# Adding remaining labels
for nerClass in ['PESSOA', 'LOCAL', 'ORGANIZACAO', 'TEMPO', 'VALOR']:
    label2Idx[nerClass] = idx
    idx += 1

#Inverse label mapping
idx2Label = {v: k for k, v in label2Idx.items()}

#Number of neurons
n_in = 2*windowSize+1
n_hidden = n_in*embeddings.shape[1]
n_out = len(label2Idx)


# Read in data
print "Read in data and create matrices"
train_sentences = PortEvalReader.readFile(trainFile)
test_sentences = PortEvalReader.readFile(testFile)


# Create numpy arrays
train_x, train_y = PortEvalReader.createNumpyArray(train_sentences, windowSize, word2Idx, label2Idx)
test_x, test_y = PortEvalReader.createNumpyArray(test_sentences, windowSize, word2Idx, label2Idx)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# Create the  Network
print "Embeddings shape",embeddings.shape

model = Sequential()
# Embeddings layers, lookups the word indices and maps them to their dense vectors. FixedEmbeddings are _not_ updated during training
# If you switch it to an Embedding-Layer, they will be updated (training time increases significant)
model.add(FixedEmbedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in,  weights=[embeddings]))
# Hidden + Softmax Layer
model.add(LSTM(output_dim=n_hidden, init='glorot_uniform', activation='tanh', batch_input_shape=(None,n_in,embeddings.shape[1]),return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(output_dim=n_hidden, init='glorot_uniform', activation='tanh',batch_input_shape=(None,n_in,n_hidden)))
model.add(Dense(output_dim=n_out, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adagrad')

model.load_weights('modelos_treinados/2-LSTM_batch200_comjanela/modelo_final.h5')
print 'Testing...'

### Testing ###
b = model.predict_classes(test_x, verbose=0)
label_y = [idx2Label[element] for element in test_y]
pred_labels = [idx2Label[element] for element in b]

print '\nPrecision, Recall, F-measure por classe (PESSOA, LOCAL, ORGANIZACAO, TEMPO, VALOR, O): '
print precision_recall_fscore_support(label_y, pred_labels, labels=['PESSOA','LOCAL','ORGANIZACAO','TEMPO', 'VALOR','O'])
print '\nF-measure Total:'
print f1_score(label_y, pred_labels, labels=['PESSOA','LOCAL','ORGANIZACAO','TEMPO', 'VALOR','O'], average='macro')