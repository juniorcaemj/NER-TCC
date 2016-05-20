import gzip

import PortEvalReader
import unittest
import numpy as np

class Testes(unittest.TestCase):

    def setUp(self):
        self.trainFile = '../corpus/corpus_First_HAREM.txt'
        self.testFile = '../corpus/corpus_miniHAREM.txt'
        vocabPath =  '../embeddings/Portuguese.vocab.gz'
        self.train_sentences, self.max_charlen = PortEvalReader.readFile2(self.trainFile)
        self.test_sentences, self.max_charlen2 = PortEvalReader.readFile2(self.testFile)
        self.windowSize = 3
        self.embeddings, self.word2Idx = self._read_word_embeddings(vocabPath)
        self.label2Idx = self._setLabel2Idx()

    def _read_word_embeddings(self, vocabPath):
        embeddings = []
        word2Idx = {}

        with gzip.open(vocabPath, 'r') as fIn:
            idx = 0
            charIdx = 1
            for line in fIn:
                split = line.strip().split(' ')
                embeddings.append(np.array([float(num) for num in split[1:]]))
                word2Idx[split[0].decode('utf-8')] = idx
                idx += 1
        return np.asarray(embeddings, dtype='float32'), word2Idx

    def _setLabel2Idx(self):
        label2Idx  = {u'O':0}
        idx = 1

        # Adding remaining labels
        for nerClass in [u'PESSOA', u'LOCAL', u'ORGANIZACAO', u'TEMPO', u'VALOR']:
            label2Idx[nerClass] = idx
            idx += 1
        return label2Idx

    def testCharEmbeddingsOnly(self):
        # Create numpy arrays
        train_x, train_y, train_x_char, char2Idx = PortEvalReader.createNumpyArrayAndCharData(self.train_sentences,
                                                                                              self.windowSize,
                                                                                              self.word2Idx,
                                                                                              self.label2Idx,
                                                                                              self.max_charlen)
        test_x, test_y, test_x_char, temp = PortEvalReader.createNumpyArrayAndCharData(self.test_sentences,
                                                                                       self.windowSize, self.word2Idx,
                                                                                       self.label2Idx,
                                                                                       self.max_charlen)

if __name__ == '__main__':
    unittest.main()
