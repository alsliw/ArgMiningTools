#!/usr/bin/env python
# encoding: utf-8

import Reader, csv, nltk, LSTM_Processor, LSTM_Main
import numpy as np, pickle
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter
from sklearn.externals import joblib
from multiprocessing import Pool

reader = Reader.readData()
docs = reader[0]
docNames = reader[1]
wnl = WordNetLemmatizer()
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
digits = '''0123456789'''
verbList = ['believe', 'think', 'agree', 'argue', 'claim', 'emphasise', 'contend', 'maintain', 
            'assert', 'theorize', 'support', 'deny', 'negate', 'refute', 'reject', 'challenge',
            'assume', 'propose', 'recommend', 'consider', 'indicate', 'observe', 'report', 'intimate']

adverbList = ['also', 'often', 'really', 'rarely', 'occasionally', 'possibly', 'probably', 'certainly', 
              'definitely', 'unquestionably', 'absolutely', 'perhaps', 'maybe', 'undoubtedly']

modalVerbs = ['shall', 'can', 'may', 'must', 'should', 'could', 'might']

argMapper = {0: 'Argument', 1: 'Non-Argument'}

''' Load Lists '''
with open("lists/unigramList.txt", "rb") as fp:   # Unpickling
    unigramList = pickle.load(fp)

with open("lists/posList.txt", "rb") as fp:   # Unpickling
    posList = pickle.load(fp)
        
''' Load transformers '''
with open("transformers/argSentTfIdfStructuralTransformer.pkl", "rb") as fp:   # Unpickling
    stT = joblib.load(fp)        
        
with open("transformers/argSentTfIdfLexicalTransformer.pkl", "rb") as fp:   # Unpickling
    leT = joblib.load(fp)

with open("transformers/argSentTfIdfSyntacticalTransformer.pkl", "rb") as fp:   # Unpickling
    syT = joblib.load(fp)   

with open("transformers/argSentTfIdfContextualTransformer.pkl", "rb") as fp:   # Unpickling
    coT = joblib.load(fp)                   
    
''' Load models '''
with open("models/lda.pkl", "rb") as fp:   # Unpickling
    lda = joblib.load(fp)    

with open("models/svm.pkl", "rb") as fp:   # Unpickling
    svc = joblib.load(fp)          

with open("models/lr.pkl", "rb") as fp:   # Unpickling
    lr = joblib.load(fp)    

with open("models/rf.pkl", "rb") as fp:   # Unpickling
    rf = joblib.load(fp)  

with open("models/ad.pkl", "rb") as fp:   # Unpickling
    ad = joblib.load(fp)    

with open("models/knn.pkl", "rb") as fp:   # Unpickling
    knn = joblib.load(fp)  

with open("models/gnb.pkl", "rb") as fp:   # Unpickling
    gnb = joblib.load(fp)                                    

def csv_reader(path):
    with open(path, 'rb') as csvfile:
        tmp = []
        reader = csv.reader(csvfile)
        for line in reader:
            if line[0] != 'model':
                tmp.append([line[0], line[1]])
        return tmp        

modelUsage = csv_reader('property.txt')
    
def csv_writer(data, path):
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)
                
def extractStructuralFeatures(s):
    auxArr = []
    covSentence = nltk.word_tokenize(s)
    covSentenceLength = covSentence.__len__()
    auxArr.append(covSentenceLength)
    auxArr.append(float(len(s))/len(covSentence))
    
    containsDigit = 0  
    for char in s:
        if char in digits:
            containsDigit = 1
            auxArr.append(1)    # contains a digit
            break
    
    if containsDigit == 0:
        auxArr.append(0)        # contains not a digit
    
    ''' #punctuations of covSentence'''
    n2 = 0
    for el in covSentence:
        if el in punctuations:
            n2 += 1
            
    auxArr.append(n2)      
    
    ''' covSentence close with question mark?'''
    if '?' in covSentence[-1]:
        auxArr.append(1)  
    else:
        auxArr.append(0)
    
    return auxArr    
        
def extractLexicalFeatures(s):
    auxArr = []
    bigr = Counter(ngrams(nltk.word_tokenize(s), 1))
    ''' does every frequent unigram occur in actual premise?'''
    for pair in unigramList:
        if pair in bigr:
            auxArr.append(1)
        else:
            auxArr.append(0) 
    
    verbBool = 0
    for v in verbList:
        if v in s.lower():
            verbBool = 1
            auxArr.append(verbBool)
            break   
        
    if verbBool == 0:
        auxArr.append(verbBool)
    
    adverbBool = 0
    for av in adverbList:
        if av in s.lower():
            adverbBool = 1
            auxArr.append(adverbBool)
            break 
          
    if adverbBool == 0:
        auxArr.append(adverbBool)                

    modverbBool = 0
    for mv in modalVerbs:
        if mv in s.lower():
            modverbBool = 1
            auxArr.append(modverbBool)
            break 
          
    if modverbBool == 0:
        auxArr.append(modverbBool)
    
    return auxArr    

def extractSyntacticalFeatures(s):
    auxArr = []      
    token = word_tokenize(s)
    pos_token = pos_tag(token)
    pos = []
    unique_pos = []
    for pt in pos_token:
        pos.append(pt[1])
        if pt[1] not in unique_pos:
            unique_pos.append(pt[1])
        
    for pos_triple in posList:
        if set(pos_triple) < set(pos):
            auxArr.append(1)
        else:
            auxArr.append(0)    
    
    future = len([word for word in pos_token if word[1] == "MD"])
    present = len([word for word in pos_token if word[1] in ["VBP", "VBZ","VBG"]])
    past = len([word for word in pos_token if word[1] in ["VBD", "VBN"]]) 
    auxArr.append(future)
    auxArr.append(present)
    auxArr.append(past) 
    
    return auxArr

#doc is a list of tokenized sentences and ind is sentence position
#example doc = ["This is first sentence", "Now the second one", "And is this the third?"]
def extractContextualFeatures(doc, ind):
    auxArr = []
    if ind != 0:
        #precedingSent = doc[ind-1]
        predToken = nltk.word_tokenize(doc[ind-1])
        predTokenLength = predToken.__len__()
        auxArr.append(predTokenLength)
        n = 0
        for el in predToken:
            if el in punctuations:
                n += 1
        
        auxArr.append(n)
        modverbBoolP = 0
        for mv in modalVerbs:
            if mv in doc[ind-1].lower():
                modverbBoolP = 1
                auxArr.append(modverbBoolP)
                break 
              
        if modverbBoolP == 0:
            auxArr.append(modverbBoolP)       
    else:
        auxArr.append(0)    
        auxArr.append(0)  
        auxArr.append(0)     
        
    if ind != (len(doc)-1)  :      
        #followingSent = doc[ind+1]
        followToken = nltk.word_tokenize(doc[ind+1])
        followTokenLength = followToken.__len__()
        auxArr.append(followTokenLength)
        m = 0
        for el in followToken:
            if el in punctuations:
                m += 1
        
        auxArr.append(m)
        modverbBoolF = 0
        for mv in modalVerbs:
            if mv in doc[ind+1].lower():
                modverbBoolF = 1
                auxArr.append(modverbBoolF)
                break 
              
        if modverbBoolF == 0:
            auxArr.append(modverbBoolF)       
    else:
        auxArr.append(0)    
        auxArr.append(0)  
        auxArr.append(0) 
    
    return auxArr 

def predict(doc):          
    sent_index = 0
    resList = []
    resList.append(docNames[doc])
    modelString = ""
    for el in modelUsage:
        if el[1] == "1":
            modelString += el[0] + ","
    firstLine = "Sentence," + modelString + "MajorityVote"  
    resList.append(firstLine.split(","))
    lstm_index = 0
    if modelUsage[-1][1] == "1":    # check lstm selection
        lstm_processor = LSTM_Processor.lstm_process("data/original/" + docNames[doc])
        lstm_predictions = LSTM_Main.lstm_main()
    for sent in docs[doc]:
        lemmatized_sent = []
        for word in word_tokenize(sent):
            try:
                lemmatized_sent.append(wnl.lemmatize(word))
            except:
                lemmatized_sent.append(word)  
        cleansed_sent = ' '.join(lemmatized_sent)          
        stFE = extractStructuralFeatures(cleansed_sent)
        leFE = extractLexicalFeatures(cleansed_sent)
        syFE = extractSyntacticalFeatures(cleansed_sent)
        coFE = extractContextualFeatures(docs[doc], sent_index)
        sent_index += 1
        struc = stT.transform([stFE]).toarray()
        lex = leT.transform([leFE]).toarray()
        synt = syT.transform([syFE]).toarray()
        cont = coT.transform([coFE]).toarray()
        feat = np.concatenate((struc, lex, synt, cont), axis=1)
        predList = []
        if modelUsage[0][1] == "1":
            predList.append(argMapper[lda.predict(feat)[0]])
        if modelUsage[1][1] == "1":    
            predList.append(argMapper[svc.predict(feat)[0]])
        if modelUsage[2][1] == "1":     
            predList.append(argMapper[lr.predict(feat)[0]])
        if modelUsage[3][1] == "1":     
            predList.append(argMapper[rf.predict(feat)[0]])
        if modelUsage[4][1] == "1":     
            predList.append(argMapper[ad.predict(feat)[0]])
        if modelUsage[5][1] == "1":     
            predList.append(argMapper[knn.predict(feat)[0]])
        if modelUsage[6][1] == "1":     
            predList.append(argMapper[gnb.predict(feat)[0]])
        if modelUsage[7][1] == "1":     
            predList.append(lstm_predictions[lstm_index])
            lstm_index += 1    
        if predList.count("Argument") == predList.count("Non-Argument"):     # in case of ties
            pred = "Undecided"
        else:        
            pred = Counter(predList).most_common(1)[0][0]
        auxList = []
        auxList.append(sent.encode('utf-8'))    # sentence
        for voter in predList:  # List of individual Votes
            auxList.append(voter)    
        auxList.append(pred)    # Majority Vote 
        resList.append(auxList)
        #print 'Class ', Counter(predList).most_common(1)[0]
    return resList    
    
def processData():
    p = Pool()
    result = p.map(predict, docs)
    p.close()
    p.join()
    return result

def writeOutput():
    res = processData()
    for r in res:
        ''' write csv file with argumentative annotations '''
        csv_writer(r[1:], "data/updated/" + r[0] + ".csv")

if __name__ == "__main__":
    writeOutput()  
    