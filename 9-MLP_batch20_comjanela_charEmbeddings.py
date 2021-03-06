# -*- coding: utf-8 -*-
import gzip
import time
from pprint import pprint

from keras.engine import Merge
from keras.layers import Embedding, Flatten, TimeDistributed
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense, Reshape
from keras.layers.recurrent import *
from keras.models import *
from keras.utils import np_utils
from sklearn.metrics import f1_score

import PortEvalReader

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

def measure(predict, groundtruth, vocab_label_size, bgn_label_idx):
    '''
        get precision, recall, f1 score
    '''
    tp = []
    fp = []
    fn = []
    recall = 0
    precision = 0
    for i in range(vocab_label_size):
        tp.append(0)
        fp.append(0)
        fn.append(0)

    for i in range(len(groundtruth)):
        if groundtruth[i] == predict[i]:
            tp[groundtruth[i]] += 1
        else:
            fp[predict[i]] += 1
            fn[groundtruth[i]] += 1

    for i in range(vocab_label_size):
        # do not count begin label
        if i == bgn_label_idx:
            continue
        if tp[i] + fp[i] == 0:
            precision += 1
        else:
            precision += float(tp[i]) / float(tp[i] + fp[i])
        if tp[i] + fn[i] == 0:
            recall += 1
        else:
            recall += float(tp[i]) / float(tp[i] + fn[i])

    precision /= (vocab_label_size - 1)
    recall /= (vocab_label_size - 1)
    pprint(tp)
    pprint(fp)
    pprint(fn)
    f1 = 2 * float(precision) * float(recall) / (precision + recall)
    print ('precision: %f, recall: %f, f1 score on testa is %f' % (precision, recall, f1))


print "Word Embeddings shape",embeddings.shape

#Word Embeddings
model_wordemb = Sequential()
model_wordemb.add(Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in,
                         weights=[embeddings]))

#Character Embeddings
model_cnn = Sequential()
model_cnn.add(Embedding(output_dim=char_embedding_size, input_dim=char_vocab_size, input_length=n_in*max_charlen_padded))
model_cnn.add(Reshape((n_in, max_charlen_padded, char_embedding_size)))
model_cnn.add(TimeDistributed(Convolution1D(nb_filter=n_filters, filter_length=char_window, border_mode='valid')))
model_cnn.add(TimeDistributed(MaxPooling1D(pool_length=max_charlen)))
model_cnn.add(Reshape((n_in, n_filters)))

'''model_cnn = Sequential()
model_cnn.add(Embedding(output_dim=char_embedding_size, input_dim=char_vocab_size, input_length=n_in*max_charlen))
model_cnn.add(Reshape((n_in, max_charlen, char_embedding_size)))
model_cnn.add(Permute((1,3,2)))
model_cnn.add(Convolution2D(nb_filter=n_filters, nb_row=1, nb_col=char_window*max_charlen, border_mode='same'))
model_cnn.add(MaxPooling2D((1, max_charlen)))
model_cnn.add(Reshape((n_in, char_embedding_size)))'''


# Create the  Network
model = Sequential()
# Hidden + Softmax Layer
model.add(Merge([model_wordemb, model_cnn], mode='concat'))
model.add(Flatten())
model.add(Dense(output_dim=n_hidden, init='glorot_uniform', activation='tanh',))
#model.add(Dropout(0.5))
model.add(Dense(output_dim=n_hidden, init='glorot_uniform', activation='tanh',))
#model.add(Dropout(0.5))
model.add(Dense(output_dim=n_out, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

from keras.utils.visualize_util import plot
plot(model, to_file='model2.png')

print train_x.shape[0], ' train samples'
print train_x.shape[1], ' train dimension'

# Train_y is a 1-dimensional vector containing the index of the label
# With np_utils.to_categorical we map it to a 1 hot matrix
train_y_cat = np_utils.to_categorical(train_y, n_out)

print "%d epochs" % number_of_epochs
print "%d mini batches" % (len(train_x)/minibatch_size)

'''for i in range(1,8):
    model.load_weights('modelos_treinados/7-LSTM_batch20_comjanela_charEmbeddings/parciais/modelo_parcial'+str(i)+'.h5')
    a = model.predict_classes([test_x, test_x_char])
    measure(a,test_y,n_out,-1)
    from sklearn.metrics import f1_score
    print 'SKLearn F1-Score: '+str(f1_score(test_y, a, average='macro'))
exit(0)'''
'''
### Testing ###
tp = 0.0
fp = 0.0
fn = 0.0

for batch in iterate_minibatches2(test_x, test_y, test_x_char, 2000, shuffle=False):
    inputs, a, inputs2 = batch
    b = model.predict([inputs,inputs2],verbose=0)
    label_y = [idx2Label[element] for element in a]
    pred_labels = [idx2Label[np.argmax(element)] for element in b]
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
    print ("Erro de divisão por zero. Continuando...")'''

'''### Testing ###
tp = 0.0
fp = 0.0
fn = 0.0
for i in range(7,8):
    print 'Modelo '+str(i)
    model.load_weights('modelos_treinados/7-LSTM_batch20_comjanela_charEmbeddings/parciais/modelo_parcial'+str(i)+'.h5')

    ### Testing ###
    tp = 0.0
    fp = 0.0
    fn = 0.0

    for batch in iterate_minibatches2(test_x, test_y, test_x_char, 2000, shuffle=False):
        inputs, a, inputs2 = batch
        b = model.predict({'input1': inputs, "input2": inputs2},verbose=0)
        label_y = [idx2Label[element] for element in a]
        pred_labels = [idx2Label[np.argmax(element)] for element in b['output']]
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
    print ' '
exit(0)'''
'''model.load_weights('modelos_treinados/7-LSTM_batch20_comjanela_charEmbeddings/parciais/modelo_parcial7.h5')
tp = 0.0
fp = 0.0
fn = 0.0
print 'Pessoa'
for batch in iterate_minibatches2(test_x, test_y, test_x_char, 2000, shuffle=False):
    inputs, a, inputs2 = batch
    b = model.predict([inputs,inputs2],verbose=0)
    label_y = [idx2Label[element] for element in a]
    pred_labels = [idx2Label[np.argmax(element)] for element in b]
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
for batch in iterate_minibatches2(test_x, test_y, test_x_char, 2000, shuffle=False):
    inputs, a, inputs2 = batch
    b = model.predict([inputs,inputs2],verbose=0)
    label_y = [idx2Label[element] for element in a]
    pred_labels = [idx2Label[np.argmax(element)] for element in b]
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
for batch in iterate_minibatches2(test_x, test_y, test_x_char, 2000, shuffle=False):
    inputs, a, inputs2 = batch
    b = model.predict([inputs,inputs2],verbose=0)
    label_y = [idx2Label[element] for element in a]
    pred_labels = [idx2Label[np.argmax(element)] for element in b]
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
for batch in iterate_minibatches2(test_x, test_y, test_x_char, 2000, shuffle=False):
    inputs, a, inputs2 = batch
    b = model.predict([inputs,inputs2],verbose=0)
    label_y = [idx2Label[element] for element in a]
    pred_labels = [idx2Label[np.argmax(element)] for element in b]
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
for batch in iterate_minibatches2(test_x, test_y, test_x_char, 2000, shuffle=False):
    inputs, a, inputs2 = batch
    b = model.predict([inputs,inputs2],verbose=0)
    label_y = [idx2Label[element] for element in a]
    pred_labels = [idx2Label[np.argmax(element)] for element in b]
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

#model.load_weights('modelos_treinados/7-LSTM_batch20_comjanela_charEmbeddings/parciais/modelo_parcial7.h5')

#Training
print 'Training...'
for epoch in xrange(number_of_epochs):
    print '\nEpoch '+str(epoch+1)+'\n'
    start_time = time.time()

    #Train for 1 epoch
    hist = model.fit([train_x, train_x_char], train_y_cat, nb_epoch=1, batch_size=minibatch_size,
                     verbose=True, shuffle=True)
    print "%.2f sec for training\n" % (time.time() - start_time)

    #Save temporary model
    model.save_weights('modelos_treinados/9-MLP_batch20_comjanela_charEmbeddings/parciais/modelo_parcial'+str(epoch+1)+'.h5',overwrite=True)

    ### Testing ###
    a = model.predict_classes([test_x, test_x_char])
    measure(a,test_y,n_out,-1)
    print 'SKLearn F1-Score: '+str(f1_score(test_y, a, average='macro'))+'\n\n'
    '''
    tp = 0.0
    fp = 0.0
    fn = 0.0
    for batch in iterate_minibatches2(test_x, test_y, test_x_char, 2000, shuffle=False):
        inputs, a, inputs2 = batch
        b = model.predict([inputs, inputs2],verbose=0)
        label_y = [idx2Label[element] for element in a]
        pred_labels = [idx2Label[np.argmax(element)] for element in b]
        for i in xrange(0,len(label_y)):
            if pred_labels[i] <> u'O' and label_y[i] <> u'O' and pred_labels[i] == label_y[i]:
                tp += 1
            elif pred_labels[i] == u'O' and label_y[i] <> u'O':
                fn += 1
            elif pred_labels[i] <> u'O' and label_y[i] <> pred_labels[i]:
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
        print ("Erro de divisão por zero. Continuando...")'''

#Save final model
model.save_weights('modelos_treinados/9-MLP_batch20_comjanela_charEmbeddings/modelo_final.h5',overwrite=True)