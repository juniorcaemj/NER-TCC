# -*- coding: utf-8 -*-

import codecs
import re

import numpy as np
from unidecode import unidecode

"""
Functions to read in the files from the GermEval contest, 
create suitable numpy matrices for train/dev/test

@author: Nils Reimers
"""


def readFile(filepath):
    sentences = []
    sentence = []
    
    for line in open(filepath):
        line = line.strip()
        
        if len(line) == 0:
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split('\t')
        sentence.append([splits[0], splits[1]])
    
    return sentences

def readFile2(filepath):
    sentences = []
    sentence = []
    max_charlen = 0
    words = []

    for line in codecs.open(filepath, 'rb', 'utf-8'):
        line = line.strip()
        if len(line) == 0:
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split('\t')
        if len(splits[0]) > max_charlen:
            max_charlen = len(splits[0])
        sentence.append([splits[0], splits[1]])
        words.append(splits[0])

    return sentences, max_charlen

def multiple_replacer(key_values):
    #replace_dict = dict(key_values)
    replace_dict = key_values
    replacement_function = lambda match: replace_dict[match.group(0)]
    pattern = re.compile("|".join([re.escape(k) for k, v in key_values.iteritems()]), re.M)
    return lambda string: pattern.sub(replacement_function, string)
    

def multiple_replace(string, key_values):
    return multiple_replacer(key_values)(string)

def normalizeWord(line):         
    try:
        line = unicode(line, "utf-8") #Convert to UTF8
    except:
        pass
    line = line.replace(u"„", u"\"")
   
    line = line.lower(); #To lower case
     
    #Replace all special charaters with the ASCII corresponding, but keep Umlaute
    #Requires that the text is in lowercase before
    replacements = dict(((u"ß", "SZ"), (u"ä", "AE"), (u"ü", "UE"), (u"ö", "OE")))
    replacementsInv = dict(zip(replacements.values(),replacements.keys()))
    line = multiple_replace(line, replacements)
    line = unidecode(line)
    line = multiple_replace(line, replacementsInv)
     
    line = line.lower() #Unidecode might have replace some characters, like € to upper case EUR
     
    line = re.sub("([0-9][0-9.,]*)", '0', line) #Replace digits by NUMBER        

   
    return line.strip();
        
def createNumpyArray(sentences, windowsize, word2Idx, label2Idx):
    unknownIdx = word2Idx['UNKNOWN']
    paddingIdx = word2Idx['PADDING']    
    
    xMatrix = []
    yVector = []
    
    wordCount = 0
    unknownWordCount = 0
    
    for sentence in sentences:
        targetWordIdx = 0
        
        for targetWordIdx in xrange(len(sentence)):
            
            # Get the context of the target word and map these words to the index in the embeddings matrix
            wordIndices = []    
            for wordPosition in xrange(targetWordIdx-windowsize, targetWordIdx+windowsize+1):
                if wordPosition < 0 or wordPosition >= len(sentence):
                    wordIndices.append(paddingIdx)
                    continue
                
                word = sentence[wordPosition][0]
                wordCount += 1
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif word.lower() in word2Idx:
                    wordIdx = word2Idx[word.lower()] 
                elif normalizeWord(word) in word2Idx:
                    wordIdx = word2Idx[normalizeWord(word)] 
                else:
                    wordIdx = unknownIdx
                    unknownWordCount += 1
                
                
                wordIndices.append(wordIdx)
                
            #Get the label and map to int
            labelIdx = label2Idx[sentence[targetWordIdx][1]]
            
            xMatrix.append(wordIndices)
            yVector.append(labelIdx)

    print "Unknowns: %.2f%%" % (unknownWordCount/(float(wordCount))*100)
    return (np.asarray(xMatrix, dtype='int32'), np.asarray(yVector, dtype='int32'))

def createNumpyArrayAndCharData(sentences, windowsize, word2Idx, label2Idx, max_charlen):
    unknownIdx = word2Idx[u'UNKNOWN']
    paddingIdx = word2Idx[u'PADDING']

    xMatrix = []
    yVector = []
    charIdxMatrix = []

    wordCount = 0
    unknownWordCount = 0

    idx = 3 # 0-padding, 1-begin_of_sentence_pad, 2-end_of_sentence_pad
    char2Idx = {}

    for sentence in sentences:
        targetWordIdx = 0

        for targetWordIdx in xrange(len(sentence)):

            # Get the context of the target word and map these words to the index in the embeddings matrix
            wordIndices = []
            charIndices = []
            for wordPosition in xrange(targetWordIdx-windowsize, targetWordIdx+windowsize+1):
                if wordPosition < 0 or wordPosition >= len(sentence):
                    wordIndices.append(paddingIdx)
                    for i in range(0,windowsize):
                        charIndices.append(1)
                    for i in range(0,(max_charlen)):
                        charIndices.append(0)
                    for j in range(0,windowsize):
                        charIndices.append(2)
                    continue

                word = sentence[wordPosition][0]
                wordCount += 1
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif word.lower() in word2Idx:
                    wordIdx = word2Idx[word.lower()]
                elif normalizeWord(word) in word2Idx:
                    wordIdx = word2Idx[normalizeWord(word)]
                else:
                    wordIdx = unknownIdx
                    unknownWordCount += 1

                for i in range(0,windowsize):
                    charIndices.append(1)
                for i in word:
                    if i not in char2Idx:
                        charIndices.append(idx)
                        char2Idx[i] = idx
                        idx += 1
                    else:
                        charIndices.append(char2Idx[i])
                while (len(charIndices) % (max_charlen+(windowsize*2))) != max_charlen+windowsize:
                    charIndices.append(0)
                for j in range(0,windowsize):
                    charIndices.append(2)
                wordIndices.append(wordIdx)

            #Get the label and map to int
            labelIdx = label2Idx[sentence[targetWordIdx][1]]

            xMatrix.append(wordIndices)
            yVector.append(labelIdx)
            charIdxMatrix.append(charIndices)

    print "Unknown Words: %.2f%%" % (unknownWordCount/(float(wordCount))*100)
    return (np.asarray(xMatrix, dtype='int32'), np.asarray(yVector, dtype='int32'), np.asarray(charIdxMatrix, dtype='int32'), char2Idx)

def createNumpyArray2(sentences, windowsize, word2Idx, label2Idx):
    unknownIdx = word2Idx['UNKNOWN']
    paddingIdx = word2Idx['PADDING']

    xMatrix = []
    yVector = []

    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        targetWordIdx = 0

        for targetWordIdx in xrange(len(sentence)):

            # Get the context of the target word and map these words to the index in the embeddings matrix
            wordIndices = []
            for wordPosition in xrange(targetWordIdx-windowsize, targetWordIdx+windowsize+1):
                if wordPosition < 0 or wordPosition >= len(sentence):
                    wordIndices.append(paddingIdx)
                    continue

                word = sentence[wordPosition]
                wordCount += 1
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif word.lower() in word2Idx:
                    wordIdx = word2Idx[word.lower()]
                elif normalizeWord(word) in word2Idx:
                    wordIdx = word2Idx[normalizeWord(word)]
                else:
                    wordIdx = unknownIdx
                    unknownWordCount += 1


                wordIndices.append(wordIdx)

            xMatrix.append(wordIndices)
    
    
    print "Unknowns: %.2f%%" % (unknownWordCount/(float(wordCount))*100)
    return (np.asarray(xMatrix, dtype='int32'))


def createNumpyArrayWithTime(sentences, windowsize, word2Idx, label2Idx, embeddings):
    unknownIdx = word2Idx['UNKNOWN']
    paddingIdx = word2Idx['PADDING']

    xMatrix = []
    yVector = []

    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        targetWordIdx = 0

        for targetWordIdx in xrange(len(sentence)):

            # Get the context of the target word and map these words to the index in the embeddings matrix
            wordIndices = []
            for wordPosition in xrange(targetWordIdx-windowsize, targetWordIdx+windowsize+1):
                if wordPosition < 0 or wordPosition >= len(sentence):
                    wordIndices.append(embeddings[paddingIdx])
                    continue

                word = sentence[wordPosition][0]
                wordCount += 1
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif word.lower() in word2Idx:
                    wordIdx = word2Idx[word.lower()]
                elif normalizeWord(word) in word2Idx:
                    wordIdx = word2Idx[normalizeWord(word)]
                else:
                    wordIdx = unknownIdx
                    unknownWordCount += 1


                wordIndices.append(embeddings[wordIdx])

            #Get the label and map to int
            labelIdx = label2Idx[sentence[targetWordIdx][1]]

            xMatrix.append(wordIndices)
            yVector.append(labelIdx)

    print "Unknowns: %.2f%%" % (unknownWordCount/(float(wordCount))*100)
    return (np.asarray(xMatrix, dtype='float32'), np.asarray(yVector, dtype='float32'))

def createNumpyArrayLSTM(sentences, word2Idx, label2Idx, embeddings):
    unknownIdx = word2Idx['UNKNOWN']
    paddingIdx = word2Idx['PADDING']

    xMatrix = []
    yVector = []

    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        targetWordIdx = 0
        wordIndices = []
        labelIdx = []

        for targetWordIdx in xrange(len(sentence)):

            # Get the context of the target word and map these words to the index in the embeddings matrix
            for wordPosition in xrange(targetWordIdx, targetWordIdx+1):
                if wordPosition < 0 or wordPosition >= len(sentence):
                    wordIndices.append(paddingIdx)
                    continue

                word = sentence[wordPosition][0]
                wordCount += 1
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif word.lower() in word2Idx:
                    wordIdx = word2Idx[word.lower()]
                elif normalizeWord(word) in word2Idx:
                    wordIdx = word2Idx[normalizeWord(word)]
                else:
                    wordIdx = unknownIdx
                    unknownWordCount += 1

                wordIndices.append(wordIdx)

            #Get the label and map to int
            labelIdx.append(label2Idx[sentence[targetWordIdx][1]])

        xMatrix.append(wordIndices)
        yVector.append(labelIdx)

    print "Unknowns: %.2f%%" % (unknownWordCount/(float(wordCount))*100)
    return xMatrix, yVector

def createNumpyArrayLSTMAndCharData(sentences, word2Idx, label2Idx, embeddings, max_charlen):
    unknownIdx = word2Idx['UNKNOWN']
    paddingIdx = word2Idx['PADDING']

    xMatrix = []
    yVector = []
    charIdxMatrix = []

    wordCount = 0
    unknownWordCount = 0

    idx = 1
    char2Idx = {}

    for sentence in sentences:
        targetWordIdx = 0
        wordIndices = []
        charIndices = []
        labelIdx = []

        for targetWordIdx in xrange(len(sentence)):

            # Get the context of the target word and map these words to the index in the embeddings matrix
            for wordPosition in xrange(targetWordIdx, targetWordIdx+1):
                if wordPosition < 0 or wordPosition >= len(sentence):
                    wordIndices.append(paddingIdx)
                    for i in range(0,max_charlen):
                        charIndices.append(1)
                    continue

                word = sentence[wordPosition][0]
                wordCount += 1
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif word.lower() in word2Idx:
                    wordIdx = word2Idx[word.lower()]
                elif normalizeWord(word) in word2Idx:
                    wordIdx = word2Idx[normalizeWord(word)]
                else:
                    wordIdx = unknownIdx
                    unknownWordCount += 1

                for i in word.decode('utf-8'):
                    if i not in char2Idx:
                        charIndices.append(idx)
                        char2Idx[i] = idx
                        idx += 1
                    else:
                        charIndices.append(char2Idx[i])
                while (len(charIndices) % max_charlen) != 0:
                    charIndices.append(1)
                wordIndices.append(wordIdx)

            #Get the label and map to int
            labelIdx.append(label2Idx[sentence[targetWordIdx][1]])

        xMatrix.append(wordIndices)
        yVector.append(labelIdx)
        charIdxMatrix.append(charIndices)

    print "Unknowns: %.2f%%" % (unknownWordCount/(float(wordCount))*100)
    return xMatrix, yVector, charIdxMatrix, char2Idx