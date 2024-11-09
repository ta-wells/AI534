#!/usr/bin/env python3

from __future__ import division # no need for python3, but just in case used w/ python2

import sys
import time
import pandas as pd
from svector import svector

def read_from(textfile):
    data = pd.read_csv(textfile)
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        yield (1 if label=="+" else -1, words.split())

def make_vector(words):
    v = svector()
    for word in words:
        v[word] += 1
    return v
    
def test(devfile, model):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        err += label * (model.dot(make_vector(words))) <= 0
    return err/i  # i is |D| now

def test_blind(blind_data,model):
    label2 = []
    for i, (label, words) in enumerate(read_from(blind_data), 1): # note 1...|D|
        if model.dot(make_vector(words))<=0:
            label2.append("-")
        else:
            label2.append("+")
    return label2


            
def train2(trainfile, devfile, epochs=5,Bias=1):
    t = time.time()
    best_err = 1.
    model = svector()
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            words.insert(0,Bias) #Added this for the Bias
            sent = make_vector(words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
        dev_err = test(devfile, model)
        best_err = min(best_err, dev_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))
    return(model) #return model to be used for testing
    

if __name__ == "__main__":
    train2(sys.argv[1], sys.argv[2], 10)
