from collections import Counter
import itertools
from collections import defaultdict
import numpy as np
import sys


class NGramTransform:
    def __init__(self,length=2,vocabulary_size=10,mood='dynamic',vocabulary=None,skip=0,plus=0):
        self.length = length
        self.vocabulary_size = vocabulary_size
        self.mood = mood
        self.skip = skip
        self.plus = plus
        if self.mood == "fixed":
            self.vocabulary = {k:v for v,k in enumerate(vocabulary)}

    def _fit_transform(self,sequence):
        if self.mood == 'dynamic':
            self.vocabulary = {}
        for syscalls in sequence:
            grams = []
            for gram in zip(*[syscalls[i:] for i in range(0,self.length)]):
                if len(gram) == 1:
                    gram = gram[0]
                grams.append(gram)
                self.add_to_vocabulary(gram)
            yield grams

    def fit_transform(self,sequence):
        if self.mood == 'dynamic':
            self.vocabulary = {}
        for syscalls in sequence:
            grams = []
            start = 0
            while start < len(syscalls) - self.length:
                gram = tuple(syscalls[start:start+self.length+self.skip:self.skip+1])
                if len(gram) == 1:
                    gram = gram[0]
                grams.append(gram)
                self.add_to_vocabulary(gram)
                start += 1
            yield grams

    def transform(self,sequence):
        for syscalls in sequence:
            grams = []
            for gram in zip(*[syscalls[i:] for i in range(0,self.length)]):
                if len(gram) == 1:
                    gram = gram[0]
                if self.in_vocabulary(gram):
                    grams.append(gram)
            yield grams

    def add_to_vocabulary(self,gram):
        if self.mood == 'dynamic':
            if gram not in self.vocabulary:
                self.vocabulary[gram] = self.vocabulary.__len__()

    def in_vocabulary(self,gram):
        if self.mood == 'fixed' or self.mood == 'dynamic':
            return gram in self.vocabulary
        else:
            return True

    def count(self,sequence):
        return Counter(sequence).items()

    def index(self,seqeuence):
        if self.mood == 'full':
            s = np.sum([(self.vocabulary_size ** i) * s for i,s in enumerate(reversed(seqeuence))])
            return s
        else:
            return self.vocabulary[seqeuence]

    def at_index(self,index):
        if self.mood == 'full':
            return index
        else:
            for k,v in self.vocabulary.items():
                if v == index:
                    return k
            return None


    def feature_size(self):
        if self.mood == 'full':
            return self.vocabulary_size ** self.length
        else:
            return len(self.vocabulary) + self.plus

    def get_vocabulary(self):
        return [p for p in itertools.product([x for x in range(0, self.vocabulary_size)],repeat=self.length)]

if __name__ == "__main__":
    v = [list(range(10)) for x in range(10)]
    ngram = NGramTransform(2,100,skip=1)
    for gram in ngram.fit_transform(v):
        print(gram)