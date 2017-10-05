#!/usr/bin/env python
# encoding: utf-8

import os, codecs#, numpy as np

def readData():
    i = 0
    articles = {}
    docNames = {}
    n = 0
    for f in os.listdir("data/original"):
        filename = "data/original/" + f
        docNames[i] = f
        articleSentences = []
        with codecs.open(filename,'r',encoding='utf8') as f:
            for line in f:
                if line != ('' or '\n'):
                    articleSentences.append(line.replace('\n','')) 
        n += len(articleSentences)            
        articles[i] = articleSentences                       
        print docNames[i]," Doc.Length:",len(articleSentences)
        #print i  
        i += 1
    
    #print "Numb. Sentences: ",n
    #print "Min Doc. Length: ", min(map(len, [articles[k] for k in articles]))
    #print "Max Doc. Length: ", max(map(len, [articles[k] for k in articles]))
    #index, value = min(enumerate(map(len, [articles[k] for k in articles])), key=operator.itemgetter(1))
    #print s[index]
    #print "Avg Doc. Length: ", np.mean(map(len, [articles[k] for k in articles])) 
    return [articles, docNames]   