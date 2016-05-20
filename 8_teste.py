# -*- coding: utf-8 -*-
import gzip

from keras.layers import Embedding, Convolution2D, Permute, Activation, MaxPooling2D
from keras.layers.core import Dropout, TimeDistributedDense, Dense, Flatten, Reshape
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
minibatch_size = 20

trainFile = 'corpus/corpus_First_HAREM.txt'
testFile = 'corpus/corpus_miniHAREM.txt'

# Word Embeddings
print "Reading word embeddings"
vocabPath =  'embeddings/Portuguese.vocab.gz'

word2Idx = {} #Maps a word to the index in the embeddings matrix
embeddings = [] #Embeddings matrix

max_charlen = 0

with gzip.open(vocabPath, 'r') as fIn:
    idx = 0
    charIdx = 1
    for line in fIn:
        split = line.strip().split(' ')
        embeddings.append(np.array([float(num) for num in split[1:]]))
        if len(split[0]) > max_charlen:
            max_charlen = len(split[0])
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

#Hyper parameters
n_in = 2*windowSize+1
n_out = len(label2Idx)
n_filters = 7
char_embedding_size = 10
n_hidden = n_in*(embeddings.shape[1]+char_embedding_size)
char_window = 3

# Read in data
print "Read in data and create matrices"
train_sentences = PortEvalReader.readFile(trainFile)
test_sentences = PortEvalReader.readFile(testFile)


# Create numpy arrays
train_x, train_y, train_x_char, char2Idx = PortEvalReader.createNumpyArrayAndCharData(train_sentences, windowSize, word2Idx, label2Idx, max_charlen)
test_x, test_y, test_x_char, temp = PortEvalReader.createNumpyArrayAndCharData(test_sentences, windowSize, word2Idx, label2Idx, max_charlen)

char_vocab_size = len(char2Idx)+1

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

#Word Embeddings
model_wordemb = Sequential()
model_wordemb.add(FixedEmbedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in,
                         weights=[embeddings]))

#Character Embeddings
model_cnn = Sequential()
model_cnn.add(Embedding(output_dim=char_embedding_size, input_dim=char_vocab_size, input_length=n_in*max_charlen))
model_cnn.add(Reshape((n_in, max_charlen, char_embedding_size)))
model_cnn.add(Permute((1,3,2)))
model_cnn.add(Convolution2D(nb_filter=n_filters, nb_row=1, nb_col=char_window, border_mode='same'))
model_cnn.add(Activation('relu'))
model_cnn.add(Permute((1,2,3)))
model_cnn.add(MaxPooling2D((1, max_charlen)))
model_cnn.add(Reshape((n_in, char_embedding_size)))


# Create the  Network
model = Graph()
#Word Embeddings
model.add_input(name='input1', input_shape=(n_in,), dtype=int)
model.add_node(model_wordemb, name="word embeddings", input="input1")
#Char Embeddings
model.add_input(name='input2', input_shape=(n_in*max_charlen,), dtype=int)
model.add_node(model_cnn, name="char embeddings", input="input2")

# Hidden + Softmax Layer
model.add_node(LSTM(output_dim=n_hidden, init='glorot_uniform', activation='tanh',
                    batch_input_shape=(None,n_in,embeddings.shape[1]+char_embedding_size),return_sequences=True),
                    name='lstm1',inputs=['word embeddings','char embeddings'])
model.add_node(Dropout(0.5),name='dropout2',input='lstm1')
model.add_node(LSTM(output_dim=n_hidden, init='glorot_uniform', activation='tanh',batch_input_shape=(None,n_in,n_hidden)),
               name='lstm2',input='dropout2')
model.add_node(Dense(output_dim=n_out, activation='softmax'),name='dense',input='lstm2')
model.add_output(name='output',input='dense')
model.compile(loss={'output':'categorical_crossentropy'}, optimizer='adagrad')

model.load_weights('modelos_treinados/7-LSTM_batch20_comjanela_charEmbeddings/parciais/modelo_parcial2.h5')
print 'Testing...'

### Testing ###
b = model.predict({'input1': test_x, "input2": test_x_char},verbose=1)
label_y = [idx2Label[element] for element in test_y]
pred_labels = [idx2Label[np.argmax(element)] for element in b['output']]

print '\nPrecision, Recall, F-measure por classe (PESSOA, LOCAL, ORGANIZACAO, TEMPO, VALOR,  O): '
print precision_recall_fscore_support(label_y, pred_labels, labels=['PESSOA','LOCAL','ORGANIZACAO','TEMPO', 'VALOR', 'O'])
print '\nF-measure Total:'
print f1_score(label_y, pred_labels, labels=['PESSOA','LOCAL','ORGANIZACAO','TEMPO', 'VALOR','O'], average='macro')