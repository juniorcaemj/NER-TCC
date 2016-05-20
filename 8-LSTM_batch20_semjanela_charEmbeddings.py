# -*- coding: utf-8 -*-
import gzip
import time

from keras.layers import Convolution2D
from keras.layers import Embedding
from keras.layers import MaxPooling2D
from keras.layers.core import Dropout, TimeDistributedDense, Reshape, Permute, Activation
from keras.layers.recurrent import *
from keras.models import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

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
max_charlen = 36
char_embedding_size = 10


# Read in data
print "Reading data and create matrices"
train_sentences = PortEvalReader.readFile(trainFile)
test_sentences = PortEvalReader.readFile(testFile)

MAX_LENGTH = 0
for i in train_sentences:
    if len(i) > MAX_LENGTH:
        MAX_LENGTH = len(i)

for i in test_sentences:
    if len(i) > MAX_LENGTH:
        MAX_LENGTH = len(i)

n_filters = MAX_LENGTH

# Create numpy arrays
train_x, train_y, train_x_char, char2Idx = PortEvalReader.createNumpyArrayLSTMAndCharData(train_sentences, word2Idx, label2Idx, embeddings, max_charlen)
test_x, test_y, test_x_char, temp = PortEvalReader.createNumpyArrayLSTMAndCharData(test_sentences, word2Idx, label2Idx, embeddings, max_charlen)

char_vocab_size = len(char2Idx)+1

#Pad Sequences
train_x = pad_sequences(train_x,value=1., padding='post',maxlen=MAX_LENGTH)
train_x_char = pad_sequences(train_x_char,value=1., padding='post',maxlen=MAX_LENGTH*max_charlen)
train_y = pad_sequences(train_y, padding='post',maxlen=MAX_LENGTH)
test_x  = pad_sequences(test_x,value=1., padding='post')
test_x_char = pad_sequences(test_x_char, value=1., padding='post')
test_y =  pad_sequences(test_y, padding='post')

#Create one-hot entity vector, e.g. [1,0,0,0,0]
train_y = np.equal.outer(train_y, np.arange(n_out)).astype(np.int32)
test_y = np.equal.outer(test_y, np.arange(n_out)).astype(np.int32)

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

def iterate_minibatches2(inputs, targets, inputs2, batchsize, shuffle=False):
    assert len(inputs) == len(targets) == len(inputs2)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], inputs2[excerpt]

# Create the  Network
print "Embeddings shape",embeddings.shape

#Word Embeddings
model_wordemb = Sequential()
model_wordemb.add(FixedEmbedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=MAX_LENGTH,
                              weights=[embeddings]))

#Character Embeddings
model_cnn = Sequential()
model_cnn.add(Embedding(output_dim=char_embedding_size, input_dim=char_vocab_size, input_length=MAX_LENGTH*max_charlen))
model_cnn.add(Reshape((MAX_LENGTH, max_charlen, char_embedding_size)))
model_cnn.add(Permute((1,3,2)))
model_cnn.add(Convolution2D(nb_filter=n_filters, nb_row=1, nb_col=max_charlen, border_mode='same'))
model_cnn.add(Activation('relu'))
model_cnn.add(MaxPooling2D((1, max_charlen)))
model_cnn.add(Reshape((MAX_LENGTH, char_embedding_size)))


# Create the  Network
model = Graph()
#Word Embeddings
model.add_input(name='input1', input_shape=(n_in,), dtype=int)
model.add_node(model_wordemb, name="word embeddings", input="input1")
#Char Embeddings
model.add_input(name='input2', input_shape=(n_in*max_charlen,), dtype=int)
model.add_node(model_cnn, name="char embeddings", input="input2")

# Hidden + Softmax Layer
model.add_node(LSTM(output_dim=MAX_LENGTH, init='glorot_uniform', activation='tanh',
                    batch_input_shape=(None,MAX_LENGTH, embeddings.shape[1]+char_embedding_size),return_sequences=True,),
                    name='lstm1',inputs=['word embeddings', 'char embeddings'])
model.add_node(Dropout(0.5), name='dropout1', input='lstm1')
model.add_node(LSTM(output_dim=MAX_LENGTH, init='glorot_uniform', activation='tanh',
                    batch_input_shape=(None,MAX_LENGTH, MAX_LENGTH),return_sequences=True,)
                    , name='lstm2', input='dropout1')
model.add_node(TimeDistributedDense(output_dim=n_out, activation='softmax'), name='dense2', input='lstm2')
model.add_output(name='output',input='dense2')
model.compile(loss={'output':'categorical_crossentropy'}, optimizer='adagrad')

print train_x.shape[0], ' train samples'
print train_x.shape[1], ' train dimension'

# Train_y is a 1-dimensional vector containing the index of the label
# With np_utils.to_categorical we map it to a 1 hot matrix
train_y_cat = np_utils.to_categorical(train_y, n_out)

print "%d epochs" % number_of_epochs
print "%d mini batches" % (len(train_x)/minibatch_size)

'''model.load_weights('modelos_treinados/8-LSTM_batch20_semjanela_charEmbeddings/parciais/modelo_parcial1.h5')

### Testing ###
tp = 0.0
fp = 0.0
fn = 0.0

for batch in iterate_minibatches2(test_x, test_y, test_x_char, 100, shuffle=False):
    inputs, a, inputs2 = batch
    b = model.predict({'input1': inputs, "input2": inputs2},verbose=1)
    label_y = []
    pred_labels = []
    for sentence in a:
        for element in sentence:
            label_y.append(idx2Label[np.argmax(element)])
    for sentence in b['output']:
        for element in sentence:
            pred_labels.append(idx2Label[np.argmax(element)])
    for i in xrange(0,len(pred_labels)):
        if pred_labels[i] <> 'O' and label_y[i] <> 'O' and pred_labels[i] == label_y[i]:
            tp += 1
        elif pred_labels[i] == 'O' and label_y[i] <> 'O':
            fn += 1
        elif pred_labels[i] <> 'O' and label_y[i] <> pred_labels[i]:
            fp += 1
try:
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    fmes = 2*(prec*rec)/(prec+rec)
    print ("True Positives: {:.1f}".format(tp))
    print ("False Positives: {:.1f}".format(fp))
    print ("False Negatives: {:.1f}".format(fn))
    print ("Precision: {:.6f}".format(prec))
    print ("Recall: {:.6f}".format(rec))
    print ("F-Measure: {:.6f}".format(fmes))
    print ("-----------------------------------------------\n")
except:
    print ("Erro de divisão por zero. Continuando...")
exit(0)'''
'''
# ### Testing ###
tp = 0.0
fp = 0.0
fn = 0.0

for batch in iterate_minibatches(test_x, test_y, 2000, shuffle=False):
    inputs, a = batch
    b = model.predict_classes(inputs, verbose=0)
    label_y = []
    pred_labels = []
    for sentence in a:
        for element in sentence:
            label_y.append(idx2Label[np.argmax(element)])
    for sentence in b:
        for element in sentence:
            pred_labels.append(idx2Label[element])
    for i in xrange(0,len(label_y)):
        if pred_labels[i] <> 'O' and label_y[i] <> 'O' and pred_labels[i] == label_y[i]:
            tp += 1
        elif pred_labels[i] == 'O' and label_y[i] <> 'O':
            fn += 1
        elif pred_labels[i] <> 'O' and label_y[i] <> pred_labels[i]:
            fp += 1
try:
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    fmes = 2*(prec*rec)/(prec+rec)
    print ("True Positives: {:.1f}".format(tp))
    print ("False Positives: {:.1f}".format(fp))
    print ("False Negatives: {:.1f}".format(fn))
    print ("Precision: {:.6f}".format(prec))
    print ("Recall: {:.6f}".format(rec))
    print ("F-Measure: {:.6f}".format(fmes))
    print ("-----------------------------------------------\n")
except:
    print ("Erro de divisão por zero. Continuando...")


tp = 0.0
fp = 0.0
fn = 0.0
print 'Pessoa'
for batch in iterate_minibatches(test_x, test_y, 2000, shuffle=False):
    inputs, a = batch
    b = model.predict_classes(inputs, verbose=0)
    label_y = []
    pred_labels = []
    for sentence in a:
        for element in sentence:
            label_y.append(idx2Label[np.argmax(element)])
    for sentence in b:
        for element in sentence:
            pred_labels.append(idx2Label[element])
    for i in xrange(0,len(label_y)):
        if pred_labels[i] == 'PESSOA' and label_y[i] == 'PESSOA':
            tp += 1
        elif pred_labels[i] == 'O' and label_y[i] == 'PESSOA':
            fn += 1
        elif pred_labels[i] == 'PESSOA' and label_y[i] <> 'PESSOA':
            fp += 1
try:
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    fmes = 2*(prec*rec)/(prec+rec)
    print ("True Positives: {:.1f}".format(tp))
    print ("False Positives: {:.1f}".format(fp))
    print ("False Negatives: {:.1f}".format(fn))
    print ("Precision: {:.6f}".format(prec))
    print ("Recall: {:.6f}".format(rec))
    print ("F-Measure: {:.6f}".format(fmes))
    print ("-----------------------------------------------\n")
except:
    print ("Erro de divisão por zero. Continuando...")

tp = 0.0
fp = 0.0
fn = 0.0
print 'Local'
for batch in iterate_minibatches(test_x, test_y, 2000, shuffle=False):
    inputs, a = batch
    b = model.predict_classes(inputs, verbose=0)
    label_y = []
    pred_labels = []
    for sentence in a:
        for element in sentence:
            label_y.append(idx2Label[np.argmax(element)])
    for sentence in b:
        for element in sentence:
            pred_labels.append(idx2Label[element])
    for i in xrange(0,len(label_y)):
        if pred_labels[i] == 'LOCAL' and label_y[i] == 'LOCAL':
            tp += 1
        elif pred_labels[i] == 'O' and label_y[i] == 'LOCAL':
            fn += 1
        elif pred_labels[i] == 'LOCAL' and label_y[i] <> 'LOCAL':
            fp += 1
try:
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    fmes = 2*(prec*rec)/(prec+rec)
    print ("True Positives: {:.1f}".format(tp))
    print ("False Positives: {:.1f}".format(fp))
    print ("False Negatives: {:.1f}".format(fn))
    print ("Precision: {:.6f}".format(prec))
    print ("Recall: {:.6f}".format(rec))
    print ("F-Measure: {:.6f}".format(fmes))
    print ("-----------------------------------------------\n")
except:
    print ("Erro de divisão por zero. Continuando...")

tp = 0.0
fp = 0.0
fn = 0.0
print 'Organização'
for batch in iterate_minibatches(test_x, test_y, 2000, shuffle=False):
    inputs, a = batch
    b = model.predict_classes(inputs, verbose=0)
    label_y = []
    pred_labels = []
    for sentence in a:
        for element in sentence:
            label_y.append(idx2Label[np.argmax(element)])
    for sentence in b:
        for element in sentence:
            pred_labels.append(idx2Label[element])
    for i in xrange(0,len(label_y)):
        if pred_labels[i] == 'ORGANIZACAO' and label_y[i] == 'ORGANIZACAO':
            tp += 1
        elif pred_labels[i] == 'O' and label_y[i] == 'ORGANIZACAO':
            fn += 1
        elif pred_labels[i] == 'ORGANIZACAO' and label_y[i] <> 'ORGANIZACAO':
            fp += 1
try:
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    fmes = 2*(prec*rec)/(prec+rec)
    print ("True Positives: {:.1f}".format(tp))
    print ("False Positives: {:.1f}".format(fp))
    print ("False Negatives: {:.1f}".format(fn))
    print ("Precision: {:.6f}".format(prec))
    print ("Recall: {:.6f}".format(rec))
    print ("F-Measure: {:.6f}".format(fmes))
    print ("-----------------------------------------------\n")
except:
    print ("Erro de divisão por zero. Continuando...")

tp = 0.0
fp = 0.0
fn = 0.0
print 'Tempo'
for batch in iterate_minibatches(test_x, test_y, 2000, shuffle=False):
    inputs, a = batch
    b = model.predict_classes(inputs, verbose=0)
    label_y = []
    pred_labels = []
    for sentence in a:
        for element in sentence:
            label_y.append(idx2Label[np.argmax(element)])
    for sentence in b:
        for element in sentence:
            pred_labels.append(idx2Label[element])
    for i in xrange(0,len(label_y)):
        if pred_labels[i] == 'TEMPO' and label_y[i] == 'TEMPO':
            tp += 1
        elif pred_labels[i] == 'O' and label_y[i] == 'TEMPO':
            fn += 1
        elif pred_labels[i] == 'TEMPO' and label_y[i] <> 'TEMPO':
            fp += 1
try:
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    fmes = 2*(prec*rec)/(prec+rec)
    print ("True Positives: {:.1f}".format(tp))
    print ("False Positives: {:.1f}".format(fp))
    print ("False Negatives: {:.1f}".format(fn))
    print ("Precision: {:.6f}".format(prec))
    print ("Recall: {:.6f}".format(rec))
    print ("F-Measure: {:.6f}".format(fmes))
    print ("-----------------------------------------------\n")
except:
    print ("Erro de divisão por zero. Continuando...")

tp = 0.0
fp = 0.0
fn = 0.0
print 'Valor'
for batch in iterate_minibatches(test_x, test_y, 2000, shuffle=False):
    inputs, a = batch
    b = model.predict_classes(inputs, verbose=0)
    label_y = []
    pred_labels = []
    for sentence in a:
        for element in sentence:
            label_y.append(idx2Label[np.argmax(element)])
    for sentence in b:
        for element in sentence:
            pred_labels.append(idx2Label[element])
    for i in xrange(0,len(label_y)):
        if pred_labels[i] == 'VALOR' and label_y[i] == 'VALOR':
            tp += 1
        elif pred_labels[i] == 'O' and label_y[i] == 'VALOR':
            fn += 1
        elif pred_labels[i] == 'VALOR' and label_y[i] <> 'VALOR':
            fp += 1
try:
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    fmes = 2*(prec*rec)/(prec+rec)
    print ("True Positives: {:.1f}".format(tp))
    print ("False Positives: {:.1f}".format(fp))
    print ("False Negatives: {:.1f}".format(fn))
    print ("Precision: {:.6f}".format(prec))
    print ("Recall: {:.6f}".format(rec))
    print ("F-Measure: {:.6f}".format(fmes))
    print ("-----------------------------------------------\n")
except:
    print ("Erro de divisão por zero. Continuando...")
exit(0)'''

#Training
print 'Training...'
for epoch in xrange(0,number_of_epochs):
    print '\nEpoch '+str(epoch+1)+'\n'
    start_time = time.time()

    #Train for 1 epoch
    hist = model.fit(data={'input1':train_x,'input2':train_x_char, 'output':train_y}, nb_epoch=1, batch_size=minibatch_size,
                     verbose=True, shuffle=True)
    print "%.2f sec for training\n" % (time.time() - start_time)

    #Save temporary model
    model.save_weights('modelos_treinados/8-LSTM_batch20_semjanela_charEmbeddings/parciais/modelo_parcial'+str(epoch+1)+'.h5',overwrite=True)

    ### Testing ###
    tp = 0.0
    fp = 0.0
    fn = 0.0

    for batch in iterate_minibatches2(test_x, test_y, test_x_char, 100, shuffle=False):
        inputs, a, inputs2 = batch
        b = model.predict({'input1': inputs, "input2": inputs2},verbose=0)
        label_y = []
        pred_labels = []
        for sentence in a:
            for element in sentence:
                label_y.append(idx2Label[np.argmax(element)])
        for sentence in b['output']:
            for element in sentence:
                pred_labels.append(idx2Label[np.argmax(element)])
        for i in xrange(0,len(pred_labels)):
            if pred_labels[i] <> 'O' and label_y[i] <> 'O' and pred_labels[i] == label_y[i]:
                tp += 1
            elif pred_labels[i] == 'O' and label_y[i] <> 'O':
                fn += 1
            elif pred_labels[i] <> 'O' and label_y[i] <> pred_labels[i]:
                fp += 1
    try:
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        fmes = 2*(prec*rec)/(prec+rec)
        print ("True Positives: {:.1f}".format(tp))
        print ("False Positives: {:.1f}".format(fp))
        print ("False Negatives: {:.1f}".format(fn))
        print ("Precision: {:.6f}".format(prec))
        print ("Recall: {:.6f}".format(rec))
        print ("F-Measure: {:.6f}".format(fmes))
        print ("-----------------------------------------------\n")
    except:
        print ("Erro de divisão por zero. Continuando...")

#Save final model
model.save_weights('modelos_treinados/3-LSTM_batch20_semjanela/modelo_final.h5')