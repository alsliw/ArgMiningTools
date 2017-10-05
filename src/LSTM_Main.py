import os
os.environ['KERAS_BACKEND']='theano'
 
import numpy as np
np.random.seed(1337)  # for reproducibility
 
from keras.preprocessing import sequence
from keras import backend as K
import cPickle
from keras.models import load_model
 
# 
def wordIdxLookup(word, word_idx_map):
    if word in word_idx_map:
        return word_idx_map[word]
     
train_sentences = []

def f1score(y_true, y_pred):
    num_tp = K.sum(y_true*y_pred)
    num_fn = K.sum(y_true*(1.0-y_pred))
    num_fp = K.sum((1.0-y_true)*y_pred)
    f1 = 2.0*num_tp/(2.0*num_tp+num_fn+num_fp)
    return f1

def lstm_main():
    
    sentences, word_embeddings, word_idx_map = cPickle.load(open("mr.p","rb"))
    print "data loaded!"
     
    for datum in sentences:
    
        words = datum['text'].split()    
        wordIndices = [wordIdxLookup(word, word_idx_map) for word in words]
        
        #print wordIndices     
        train_sentences.append(wordIndices)   
             
    X_train = sequence.pad_sequences(train_sentences, maxlen=89)
    
    model = load_model('models/lstm.h5', custom_objects={'f1score': f1score}) 
    targetPredict = model.predict(X_train);
    # print('successful target', targetPredict.shape)
    result = []
    for temp in targetPredict:
        if temp > 0.5:
            result.append("Argument")
        else:
            result.append("Non-Argument")
    return result

