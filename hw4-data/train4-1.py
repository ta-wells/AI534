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

def P4_sort(devfile,model):
    tot, err = 0, 0
    wrong_list_neg =[]
    wrong_list_pos = []
    i_list_neg = []
    i_list_pos = []
    l_pos = []
    l_neg = []
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        if model.dot(make_vector(words))<=0:
            err += label * (model.dot(make_vector(words)))
            if label<0:
                i_list_neg.append(model.dot(make_vector(words)))
                wrong_list_neg.append(words) 
                
            else:
                i_list_pos.append(model.dot(make_vector(words)))
                wrong_list_pos.append(words)
                
    return i_list_neg,wrong_list_neg,i_list_pos,wrong_list_pos  # i is |D| now
            
def train4(trainfile, devfile, epochs=5,Bias=1):
    t = time.time()
    c=0
    best_err = 1.
    model = svector()
    model_av = svector() #Initialize
    
    for it in range(1, epochs+1):
        updates = 0
        
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            words.insert(0,Bias) #Added this for the Bias
            sent = make_vector(words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
                #ws = ws + w
                model_av = model_av + c*label*sent
            c = c+1
        dev_err = test(devfile, c*model-model_av)
        best_err = min(best_err, dev_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(c*model-model_av), time.time() - t))
    return(c*model-model_av) #return model to be used for testing
    

if __name__ == "__main__":
    train4(sys.argv[1], sys.argv[2], 10)
