# -*- coding: utf-8 -*-
import gzip
import time

from keras.layers.core import Dropout, Dense
from keras.layers.recurrent import *
from keras.models import *
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

print train_x.shape[0], ' train samples'
print train_x.shape[1], ' train dimension'

# Train_y is a 1-dimensional vector containing the index of the label
# With np_utils.to_categorical we map it to a 1 hot matrix
train_y_cat = np_utils.to_categorical(train_y, n_out)

print "%d epochs" % number_of_epochs
print "%d mini batches" % (len(train_x)/minibatch_size)

#model.load_weights('modelos_treinados/1-LSTM_batch20_comjanela/modelo_final.h5')

# # ### Testing ###
# tp = 0.0
# fp = 0.0
# fn = 0.0
#
# for batch in iterate_minibatches(test_x, test_y, 2000, shuffle=False):
#     inputs, a = batch
#     b = model.predict_classes(inputs, verbose=0)
#     label_y = [idx2Label[element] for element in a]
#     pred_labels = [idx2Label[element] for element in b]
#     for i in xrange(0,len(label_y)):
#         if pred_labels[i] <> 'O' and label_y[i] <> 'O' and pred_labels[i] == label_y[i]:
#             tp += 1
#         elif pred_labels[i] == 'O' and label_y[i] <> 'O':
#             fn += 1
#         elif pred_labels[i] <> 'O' and label_y[i] <> pred_labels[i]:
#             fp += 1
# try:
#     prec = tp/(tp+fp)
#     rec = tp/(tp+fn)
#     fmes = 2*(prec*rec)/(prec+rec)
#     print ("True Positives: {:.1f}".format(tp))
#     print ("False Positives: {:.1f}".format(fp))
#     print ("False Negatives: {:.1f}".format(fn))
#     print ("Precision: {:.6f}".format(prec))
#     print ("Recall: {:.6f}".format(rec))
#     print ("F-Measure: {:.6f}".format(fmes))
#     print ("-----------------------------------------------\n")
# except:
#     print ("Erro de divisão por zero. Continuando...")
# exit(0)
#
#
# tp = 0.0
# fp = 0.0
# fn = 0.0
# print 'Pessoa'
# for batch in iterate_minibatches(test_x, test_y, 2000, shuffle=False):
#     inputs, a = batch
#     b = model.predict_classes(inputs, verbose=0)
#     label_y = [idx2Label[element] for element in a]
#     pred_labels = [idx2Label[element] for element in b]
#     for i in xrange(0,len(label_y)):
#         if pred_labels[i] == 'PESSOA' and label_y[i] == 'PESSOA':
#             tp += 1
#         elif pred_labels[i] == 'O' and label_y[i] == 'PESSOA':
#             fn += 1
#         elif pred_labels[i] == 'PESSOA' and label_y[i] <> 'PESSOA':
#             fp += 1
# try:
#     prec = tp/(tp+fp)
#     rec = tp/(tp+fn)
#     fmes = 2*(prec*rec)/(prec+rec)
#     print ("True Positives: {:.1f}".format(tp))
#     print ("False Positives: {:.1f}".format(fp))
#     print ("False Negatives: {:.1f}".format(fn))
#     print ("Precision: {:.6f}".format(prec))
#     print ("Recall: {:.6f}".format(rec))
#     print ("F-Measure: {:.6f}".format(fmes))
#     print ("-----------------------------------------------\n")
# except:
#     print ("Erro de divisão por zero. Continuando...")
#
# tp = 0.0
# fp = 0.0
# fn = 0.0
# print 'Local'
# for batch in iterate_minibatches(test_x, test_y, 2000, shuffle=False):
#     inputs, a = batch
#     b = model.predict_classes(inputs, verbose=0)
#     label_y = [idx2Label[element] for element in a]
#     pred_labels = [idx2Label[element] for element in b]
#     for i in xrange(0,len(label_y)):
#         if pred_labels[i] == 'LOCAL' and label_y[i] == 'LOCAL':
#             tp += 1
#         elif pred_labels[i] == 'O' and label_y[i] == 'LOCAL':
#             fn += 1
#         elif pred_labels[i] == 'LOCAL' and label_y[i] <> 'LOCAL':
#             fp += 1
# try:
#     prec = tp/(tp+fp)
#     rec = tp/(tp+fn)
#     fmes = 2*(prec*rec)/(prec+rec)
#     print ("True Positives: {:.1f}".format(tp))
#     print ("False Positives: {:.1f}".format(fp))
#     print ("False Negatives: {:.1f}".format(fn))
#     print ("Precision: {:.6f}".format(prec))
#     print ("Recall: {:.6f}".format(rec))
#     print ("F-Measure: {:.6f}".format(fmes))
#     print ("-----------------------------------------------\n")
# except:
#     print ("Erro de divisão por zero. Continuando...")
#
# tp = 0.0
# fp = 0.0
# fn = 0.0
# print 'Organização'
# for batch in iterate_minibatches(test_x, test_y, 2000, shuffle=False):
#     inputs, a = batch
#     b = model.predict_classes(inputs, verbose=0)
#     label_y = [idx2Label[element] for element in a]
#     pred_labels = [idx2Label[element] for element in b]
#     for i in xrange(0,len(label_y)):
#         if pred_labels[i] == 'ORGANIZACAO' and label_y[i] == 'ORGANIZACAO':
#             tp += 1
#         elif pred_labels[i] == 'O' and label_y[i] == 'ORGANIZACAO':
#             fn += 1
#         elif pred_labels[i] == 'ORGANIZACAO' and label_y[i] <> 'ORGANIZACAO':
#             fp += 1
# try:
#     prec = tp/(tp+fp)
#     rec = tp/(tp+fn)
#     fmes = 2*(prec*rec)/(prec+rec)
#     print ("True Positives: {:.1f}".format(tp))
#     print ("False Positives: {:.1f}".format(fp))
#     print ("False Negatives: {:.1f}".format(fn))
#     print ("Precision: {:.6f}".format(prec))
#     print ("Recall: {:.6f}".format(rec))
#     print ("F-Measure: {:.6f}".format(fmes))
#     print ("-----------------------------------------------\n")
# except:
#     print ("Erro de divisão por zero. Continuando...")
#
# tp = 0.0
# fp = 0.0
# fn = 0.0
# print 'Tempo'
# for batch in iterate_minibatches(test_x, test_y, 2000, shuffle=False):
#     inputs, a = batch
#     b = model.predict_classes(inputs, verbose=0)
#     label_y = [idx2Label[element] for element in a]
#     pred_labels = [idx2Label[element] for element in b]
#     for i in xrange(0,len(label_y)):
#         if pred_labels[i] == 'TEMPO' and label_y[i] == 'TEMPO':
#             tp += 1
#         elif pred_labels[i] == 'O' and label_y[i] == 'TEMPO':
#             fn += 1
#         elif pred_labels[i] == 'TEMPO' and label_y[i] <> 'TEMPO':
#             fp += 1
# try:
#     prec = tp/(tp+fp)
#     rec = tp/(tp+fn)
#     fmes = 2*(prec*rec)/(prec+rec)
#     print ("True Positives: {:.1f}".format(tp))
#     print ("False Positives: {:.1f}".format(fp))
#     print ("False Negatives: {:.1f}".format(fn))
#     print ("Precision: {:.6f}".format(prec))
#     print ("Recall: {:.6f}".format(rec))
#     print ("F-Measure: {:.6f}".format(fmes))
#     print ("-----------------------------------------------\n")
# except:
#     print ("Erro de divisão por zero. Continuando...")
#
# tp = 0.0
# fp = 0.0
# fn = 0.0
# print 'Valor'
# for batch in iterate_minibatches(test_x, test_y, 2000, shuffle=False):
#     inputs, a = batch
#     b = model.predict_classes(inputs, verbose=0)
#     label_y = [idx2Label[element] for element in a]
#     pred_labels = [idx2Label[element] for element in b]
#     for i in xrange(0,len(label_y)):
#         if pred_labels[i] == 'VALOR' and label_y[i] == 'VALOR':
#             tp += 1
#         elif pred_labels[i] == 'O' and label_y[i] == 'VALOR':
#             fn += 1
#         elif pred_labels[i] == 'VALOR' and label_y[i] <> 'VALOR':
#             fp += 1
# try:
#     prec = tp/(tp+fp)
#     rec = tp/(tp+fn)
#     fmes = 2*(prec*rec)/(prec+rec)
#     print ("True Positives: {:.1f}".format(tp))
#     print ("False Positives: {:.1f}".format(fp))
#     print ("False Negatives: {:.1f}".format(fn))
#     print ("Precision: {:.6f}".format(prec))
#     print ("Recall: {:.6f}".format(rec))
#     print ("F-Measure: {:.6f}".format(fmes))
#     print ("-----------------------------------------------\n")
# except:
#     print ("Erro de divisão por zero. Continuando...")
# exit(0)

#Training
print 'Training...'
for epoch in xrange(number_of_epochs):
    print '\nEpoch '+str(epoch+1)+'\n'
    start_time = time.time()

    #Train for 1 epoch
    hist = model.fit(train_x, train_y_cat, nb_epoch=1, batch_size=minibatch_size, verbose=True, shuffle=True,show_accuracy=False,)
    print "%.2f sec for training\n" % (time.time() - start_time)
    ### Testing ###
    tp = 0.0
    fp = 0.0
    fn = 0.0

    for batch in iterate_minibatches(test_x, test_y, 2000, shuffle=False):
        inputs, a = batch
        b = model.predict_classes(inputs, verbose=0)
        label_y = [idx2Label[element] for element in a]
        pred_labels = [idx2Label[element] for element in b]
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

    #Save temporary model
    model.save_weights('modelos_treinados/1-LSTM_batch20_comjanela/parciais/modelo_parcial'+str(epoch+1)+'.h5',overwrite=True)

#Save final model
model.save_weights('modelos_treinados/1-LSTM_batch20_comjanela/modelo_final.h5',overwrite=True)

