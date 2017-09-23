# -*- coding: utf-8 -*-
'''

@author: Hieu C. Huynh
@email: hieuhc@hiof.no
'''

from nltk.stem.snowball import SnowballStemmer
from nltk import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import re
import pycrfsuite
import pickle
import os



SEED = 2015

init_features_list = ['word', 'word.shape','word.infreqterm','word.substring.3','word.substring.4','word.substring.5','word.substring.6',
'-1.word','-1|0.word','-1.word.shape','-1|0.word.shape','-1.word|0.shape','-1.word.substring.3','-1.word.substring.4','-1.word.substring.5',
'-2.word','-2.word.shape','-2.word.substring.3','-2.word.substring.4','-2.word.substring.5',
'-3.word.shape','-3.word.substring.3','-3.word.substring.5','-3.word.substring.6',
'+1.word','0|+1.word','+1.word.shape','0|+1.word.shape','0.shape|+1.word','+1.word.substring.3','+1.word.substring.4','+1.word.substring.5','+1.word.substring.6',
'+2.word','+2.word.shape','+2.word.substring.3','+2.word.substring.6',
'+3.word.shape','+3.word.substring.3','+3.word.substring.5',
'-1|0|1.word','-1|0|1.word.shape', '-1.word.numbers', 'word.comma',
'word.numbers', 'word.initCap', 'word.exist',
'-1.word.numbers','-1.word.initCap', '-1.word.exist',
'-2.word.numbers','-2.word.initCap', '-2.word.exist',
'-3.word.numbers','-3.word.initCap', '-3.word.exist',
'+1.word.numbers', '+1.word.initCap', '+1.word.exist',
'+2.word.numbers', '+2.word.initCap', '+2.word.exist',
'+3.word.numbers', '+3.word.initCap', '+3.word.exist']
model_id = 'crf3_en_1'

def frequent_terms_list (X, freq_threshold, lang = 'norwegian'):
    vectorizer = CountVectorizer(ngram_range= (1,1),  token_pattern='(?u)\\b\\w+\\b')
    vectorizer.fit(X)
    feature_name = vectorizer.get_feature_names()
    
    nor_stem = SnowballStemmer(lang)
    feature_name_stem = set([nor_stem.stem(word) for word in feature_name])
    
    word_stem_count = {word_stem : 0 for word_stem in feature_name_stem} 
    for X_string in X:        
        _ = vectorizer.fit([X_string])
        feature_name_sent = vectorizer.get_feature_names()        
        for word in feature_name_sent:
            word_stem = nor_stem.stem(word)                 
            word_stem_count[word_stem] += 1
            
    frequent_terms = []
    for w in sorted (word_stem_count, key = word_stem_count.get, reverse= True):
        if word_stem_count[w] > freq_threshold and w != 'ike':
            frequent_terms.append(w)
    return frequent_terms

    
    
def prob_2_string (X_string, y_tag_pred, y_B_prob_pred, mode):                
        tokens_list = wordpunct_tokenize(X_string)                        
        candidate_res = []
        candidate_prob = []
        res_string =''        
        res_prob = 0        
        special_char = False   
#         print(y_B_prob_pred[:5])     
        for j in range(len(tokens_list)  + 1):
            if (j == len(tokens_list) or y_tag_pred[j] =="B-beg" or y_tag_pred[j] =="M-beg"  or y_tag_pred[j] =="L-beg") and res_string != "":
                candidate_res.append(res_string)
                candidate_prob.append(res_prob)
                res_string = ""
            if j == len(tokens_list): continue
            if y_tag_pred[j] == mode + "-beg":
                res_string = tokens_list[j]
                res_prob = y_B_prob_pred[j]
            if y_tag_pred[j] == mode + "-in":
                if res_string !="":
                    if re.match('\W', tokens_list[j]):
                        res_string += tokens_list[j]
                        special_char = True
                    elif special_char:
                        res_string += tokens_list[j]
                        special_char = False
                    else:
                        res_string += ' ' + tokens_list[j]
#                 elif j + 1 < len(tokens_list) and y_tag_pred[j + 1] == 'I':
#                     res_string = tokens_list[j]
        candidate_string_list = []    
        pred_string = ""
        for k in range(len(candidate_res)):
            candidate_string_list.append(candidate_res[k])
                    
        num_max = 0                    
        for candidate_string in candidate_string_list:
            num_words = len(re.split(' ', candidate_string))
            if num_words > num_max:
                pred_string = candidate_string
                num_max = num_words
        if num_max != 0: 
#                 pred_string = candidate_string_list[-1]                
            max_val = np.amax(candidate_prob)
            max_index = 0
            for j,val in enumerate(candidate_prob):
                if val == max_val:
                    max_index = j
                    break;
            pred_string = candidate_string_list[max_index]
            max_prob = max_val
            choice = 1            
        if num_max == 0: 
            pred_string = tokens_list[0]
            max_val = np.amax(y_B_prob_pred)
            max_index = 0
            for j,val in enumerate(y_B_prob_pred):
                if val == max_val:
                    max_index = j
                    break;
            pred_string = tokens_list[max_index]            
            max_prob = max_val
            choice = 0 
            #pred_string = ""
                        
        return  max_prob, choice, pred_string     
        

def sub_string_features_1(token):
    features = []
    substr = ''
    for i in range(len(token)):
        substr = substr + token[i]
        features.append(str(i + 1) + '.substr=' + substr)
    j = len(token) - 1
    substr = ''
    while j>= 0:
        substr = token[j] + substr
        features.append('-' + str(len(token) - j) + '.substr=' + substr)
        j = j - 1
#     print (features)
    return features
def sub_string_features_2 ( token, pos, n_gram):
    if len(token) < n_gram:
        return []
    features = []
    for i in range(len(token) - n_gram):
        substr = token[i:i+n_gram]
        features.append(str(i+1) + '.' + str(n_gram) + '.' + str(pos) + 'word.substr=' + substr)
        features.append('-' + str(len(token) - n_gram + 1 - i) + '.' + str(n_gram) + '.' + str(pos) + 'word.substr=' + substr)
#     print(features)
    return features
def token_shape(token):  
    lower_regex = '[a-zåøæóôö]'
    upper_regex = '[A-ZÅØ]' 
    mix_regex = '[a-zåøæóôöA-ZÅØ]'  
    res= ''
    if not re.match('\w+', token):
        res = 'pu'
    elif token.isdigit():
        res = 'd'
        
    elif re.compile('^' + lower_regex + '+$').match(token):
        res = 'x'
        
#         if len(token) <= 2:
#             res += str(len(token))
    elif re.compile('^' + upper_regex + '+$').match(token):
        res = 'X'
          
    elif re.compile('^' + mix_regex + '*' + upper_regex + '+' + mix_regex + '*$').match( token):
        res = 'Xx'
        
    elif re.compile('^[0-9]+' + mix_regex + '+$').match( token):
        temp = re.search(mix_regex + '+', token)
        word = token[temp.start():temp.end()]
        res = 'd_' + token_shape(word)
        
    elif re.compile('^' + mix_regex + '+[0-9]+$').match (token):
        temp = re.search(mix_regex + '+', token)
        word = token[temp.start():temp.end()]
        res = token_shape(word) + '_d'
        
    elif re.compile('^[0-9]+' + mix_regex + '+[0-9]+$').match( token):
        temp = re.search(mix_regex + '+', token)
        word = token[temp.start():temp.end()]
        res = 'd_' + token_shape(word) + '_d'
        
    elif re.compile('^' + mix_regex + '+[0-9]+' + mix_regex +'+$').match (token):
        temp = re.search('[0-9]+', token)
        word1 = token[:temp.start()]
        word2 = token[temp.end() : ]
        res = token_shape(word1) + '_d_' + token_shape(word2)
    else:
        res = 'ex'    
#     print('(%s, %s)' % (token, res))
#     if res ==  'ex': print('token shape: cant find shape %s' % token)
    return res   

def read_words_list (words_file):
    words = pd.read_csv(words_file, header = None)
    return list(words.ix[:,0].values)
def list_look_up(words_list, word):
    if word in words_list:
        return 'True'
    else:
        return 'False'
        
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)
def init_cap(inputString):
    if inputString[0].isupper():
        return 1
    elif inputString.isupper():
        return 2
    else:
        return 0

def token_2_features(tokens_list, i, frequent_terms, features_list, words_list):
    word = tokens_list[i]        
    word_stem = SnowballStemmer('norwegian').stem(word)
    
    features = []
    if 'word' in features_list:
        features.append('word=' + word_stem)
        features.append('word=' + word_stem)
        features.append('word=' + word_stem)   
    if 'word.numbers' in features_list:
        features.append( 'word.numbers=' + str(hasNumbers(word_stem)))
    if 'word.initCap' in features_list:
        features.append( 'word.initCap=' + str(init_cap(word)))
    if 'word.exist' in features_list:
        features.append( 'word.exist=' + list_look_up(words_list, word))             
    if 'word.shape' in features_list:
        features.append ('word.shape=' + token_shape(word))
    if 'word.infreqterm' in features_list:
        test = word_stem in frequent_terms
        features.append('word.infreqterm=' + str(test))
        features.append('word.infreqterm=' + str(test))

    if 'word.substring.3' in features_list:        
        features.extend( sub_string_features_2(word_stem, 0, 3))
    if 'word.substring.4' in features_list:
        features.extend( sub_string_features_2(word_stem, 0, 4))
    if 'word.substring.5' in features_list:
        features.extend( sub_string_features_2(word_stem, 0, 5))
    if 'word.substring.6' in features_list:
        features.extend( sub_string_features_2(word_stem, 0, 6))
                
     
    if i > 0:
        word1 = tokens_list[i-1]
        word1_stem = SnowballStemmer('norwegian').stem(word1)
        if '-1.word.numbers' in features_list:
            features.append( '-1.word.numbers=' + str(hasNumbers(word1)))
        if '-1.word.initCap' in features_list:
            features.append( '-1.word.initCap=' + str(init_cap(word1)))
        if '-1.word.exist' in features_list:
            features.append( '-1.word.exist=' + list_look_up(words_list, word1))
        if '-1.word' in features_list:
            features.append('-1.word=' + word1_stem)            
        if '-1|0.word' in features_list:
            features.append('-1|0.word=' + word1_stem +'|' + word_stem)
        if '-1.word.shape' in features_list:
            features.append('-1.word.shape=' + token_shape(word1))
        if '-1|0.word.shape' in features_list:
            features.append('-1|0.word.shape=' + token_shape(word1) + '|' + token_shape(word))
        if '-1.word|0.shape' in features_list:
            features.append('-1.word|0.shape=' + word1_stem + '|' + token_shape(word))
        if '-1.word.substring.3' in features_list:
            features.extend(sub_string_features_2(word1_stem, -1, 3))
        if '-1.word.substring.4' in features_list:
            features.extend(sub_string_features_2(word1_stem, -1, 4))
        if '-1.word.substring.5' in features_list:
            features.extend(sub_string_features_2(word1_stem, -1, 5))
        if '-1.word.substring.6' in features_list:
            features.extend(sub_string_features_2(word1_stem, -1, 6))                    
    else:
        features.append('BOS')
      
    if i > 1:
        word2 = tokens_list[i-2]
        word2_stem = SnowballStemmer('norwegian').stem(word2)
        if '-2.word' in features_list:
            features.append('-2.word=' + word2_stem)
        if '-2.word.numbers' in features_list:
            features.append( '-2.word.numbers=' + str(hasNumbers(word2)))
        if '-2.word.initCap' in features_list:
            features.append( '-2.word.initCap=' + str(init_cap(word2)))
        if '-2.word.exist' in features_list:
            features.append( '-2.word.exist=' + list_look_up(words_list, word2))
        if  '-2.word.shape' in features_list:
            features.append('-2.word.shape=' + token_shape(word2))
        if '-2.word.substring.3' in features_list:
            features.extend(sub_string_features_2(word2_stem, -2, 3))
        if '-2.word.substring.4' in features_list:
            features.extend(sub_string_features_2(word2_stem, -2, 4))  
        if '-2.word.substring.5' in features_list:
            features.extend(sub_string_features_2(word2_stem, -2, 5))  
        if '-2.word.substring.6' in features_list:
            features.extend(sub_string_features_2(word2_stem, -2, 6))          
        
    if i > 2:
        word3 = tokens_list[i-3]
        word3_stem = SnowballStemmer('norwegian').stem(word3)  
        if '-3.word' in features_list:
            features.append( '-3.word=' + word3_stem)
        if '-3.word.numbers' in features_list:
            features.append( '-3.word.numbers=' + str(hasNumbers(word3)))
        if '-3.word.initCap' in features_list:
            features.append( '-3.word.initCap=' + str(init_cap(word3)))
        if '--3.word.exist' in features_list:
            features.append( '-3.word.exist=' + list_look_up(words_list, word3))
        if '-3.word.shape' in features_list:
            features.append('-3.word.shape=' + token_shape(word3))
        if  '-3.word.substring.3' in features_list:
            features.extend(sub_string_features_2(word3_stem, -3, 3))
        if  '-3.word.substring.4' in features_list:
            features.extend(sub_string_features_2(word3_stem, -3, 4))
        if  '-3.word.substring.5' in features_list:
            features.extend(sub_string_features_2(word3_stem, -3, 5))
        if  '-3.word.substring.6' in features_list:
            features.extend(sub_string_features_2(word3_stem, -3, 6))                
      
    if i < len(tokens_list)-1:
        word1 = tokens_list[i+1]        
        word1_stem = SnowballStemmer('norwegian').stem(word1)
        if '+1.word.numbers' in features_list:
            features.append('+1.word.numbers=' + str(hasNumbers(word1_stem)))            
        if '+1.word.initCap' in features_list:            
            features.append( '+1.word.initCap=' + str(init_cap(word1)))
        if '+1.word.exist' in features_list:
            features.append( '+1.word.exist=' + list_look_up(words_list, word1))
        if '+1.word' in features_list:
            features.append('1.word=' + word1_stem)            
        if '0|+1.word' in features_list:
            features.append('0|+1.word=' + word_stem + '|' + word1_stem)
        if '+1.word.shape' in features_list:
            features.append('+1.word.shape=' + token_shape(word1))
        if '0|+1.word.shape' in features_list:
            features.append( '0|+1.word.shape='+ token_shape(word) + '|' + token_shape(word1))
        if '0.shape|+1.word' in features_list:
            features.append('0.shape|+1.word' + token_shape(word) + '|' + word1_stem,   )
        if '+1.word.substring.3' in features_list:
            features.extend(sub_string_features_2(word1_stem, 1, 3))
        if '+1.word.substring.4' in features_list:
            features.extend(sub_string_features_2(word1_stem, 1, 4))
        if '+1.word.substring.5' in features_list:
            features.extend(sub_string_features_2(word1_stem, 1, 5))
        if '+1.word.substring.6' in features_list:
            features.extend(sub_string_features_2(word1_stem, 1, 6))       
         
    else:
        features.append('EOS')
    
    if i < len(tokens_list) - 2:
        word2 = tokens_list[i+2]
        word2_stem = SnowballStemmer('norwegian').stem(word2)
        if '+2.word.numbers' in features_list:
            features.append( '+2.word.numbers=' + str(hasNumbers(word2)))
        if '+2.word.initCap' in features_list:
            features.append( '+2.word.initCap=' + str(init_cap(word2)))
        if '+2.word.exist' in features_list:
            features.append( '+2.word.exist=' + list_look_up(words_list, word2))
        if '+2.word' in features_list:
            features.append('+2.word=' + word2_stem)
        if '+2.word.shape' in features_list:
            features.append ('+2.word.shape=' + token_shape(word2))  
        if '+2.word.substring.3' in features_list:
            features.extend(sub_string_features_2(word2_stem, 2, 3))
        if '+2.word.substring.4' in features_list:
            features.extend(sub_string_features_2(word2_stem, 2, 4))  
        if '+2.word.substring.5' in features_list:
            features.extend(sub_string_features_2(word2_stem, 2, 5))  
        if '+2.word.substring.6' in features_list:
            features.extend(sub_string_features_2(word2_stem, 2, 6))              
        
    if i < len(tokens_list) - 3:
        word3 = tokens_list[i+3]
        word3_stem = SnowballStemmer('norwegian').stem(word3)
        if '+3.word.numbers' in features_list:   
            features.append( '+3.word.numbers=' + str(hasNumbers(word3)))
        if '+3.word.initCap' in features_list:
            features.append( '+3.word.initCap=' + str(init_cap(word3)))
        if '+3.word.exist' in features_list:
            features.append( '+3.word.exist=' + list_look_up(words_list, word3))
        if '+3.word' in features_list:
            features.append( '+3.word=' + word3_stem)
        if '+3.word.shape' in features_list:
            features.append('+3.word.shape=' + token_shape(word3))
        if  '+3.word.substring.3' in features_list:
            features.extend(sub_string_features_2(word3_stem, 3, 3))
        if  '+3.word.substring.4' in features_list:
            features.extend(sub_string_features_2(word3_stem, 3, 4))
        if  '+3.word.substring.5' in features_list:
            features.extend(sub_string_features_2(word3_stem, 3, 5))
        if  '+3.word.substring.6' in features_list:
            features.extend(sub_string_features_2(word3_stem, 3, 6))              
    
    if i > 0 and i < len(tokens_list)-1 :            
        word_1 = tokens_list[i-1]
        word1 = tokens_list[i+1]
        word_1_stem = SnowballStemmer('norwegian').stem(word_1)
        word1_stem = SnowballStemmer('norwegian').stem(word1)
        if '-1|0|1.word' in features_list:
            features.append('-1|0|1.word=' + word_1_stem + '|' + word_stem + '|' + word1_stem)
        if '-1|0|1.word.shape' in features_list:
            features.append('-1|0|1.word.shape=' + token_shape(word_1) + '|' + token_shape(word) + '|' + token_shape(word1))        
    return features
def desc_2_feature(X_string, y_string, frequent_terms, features_list):
    tokens_list = wordpunct_tokenize(X_string)
    y = np.array([])
    if len(y_string) > 0:
        y_tokens = wordpunct_tokenize(y_string)
        y = np.array(range(len(tokens_list)), dtype=str).reshape(len(tokens_list))
        pos_start = -1
        pos_end = -1
        for i in range(len(tokens_list)):
            if(tokens_list[i] == y_tokens[0]):
                alike = True
                k = i
                for j in range(len(y_tokens)): 
                    if re.match('\W', tokens_list[k+j]):
                        k = k+1           
                    if tokens_list[k + j] != y_tokens[j]:
                        alike = False
                        break;            
                if alike:
                    pos_start = i
                    pos_end = k + len(y_tokens) - 1
                    break;
        for i in range(len(y)):
            if i == pos_start:
                y[i] = 'B'
            elif i > pos_start and i <= pos_end:
                y[i] = 'I' 
            else:
                y[i] = 'O'
            
    desc_features = [token_2_features(tokens_list, i,frequent_terms, features_list) for i in range(len(tokens_list))]
#     if(pos_start == -1): print('desc_2_feature: cant find (%s, %s)' % (tokens_list, y_string))
#     print(str(tokens_list).encode('utf8'))
#     print(y)
    return desc_features, y
    
def findLocationSubString(tokens_list, y_tokens):
    if len(y_tokens) > 0:
        pos_start = -1
        pos_end = -1
        for i in range(len(tokens_list)):
            if(tokens_list[i] == y_tokens[0]):
                alike = True
                k = i
                for j in range(len(y_tokens)): 
#                    if re.match('\W', tokens_list[k+j]):
#                        k = k+1           
                    if tokens_list[k + j] != y_tokens[j]:
                        alike = False
                        break;            
                if alike:
                    pos_start = i
                    pos_end = k + len(y_tokens) - 1
                    break;
    return pos_start, pos_end
    
def desc_2_tag(y, tokens_list, y_string, mode):    
    if str(y_string) == 'nan':
        y_string = ""
    pos_start = -1
    pos_end = -1
    if len(y_string) > 0:
        y_tokens = wordpunct_tokenize(y_string)
        pos_start, pos_end = findLocationSubString(tokens_list, y_tokens)
    for i in range(len(y)):
            if i == pos_start:
                if y[i] == "NA": 
                    y[i] = mode + "-beg"
                else:
                    print(y_string + " duplicated!")
            elif i > pos_start and i <= pos_end:
                if y[i] == "NA":
                    y[i] = mode + "-in" 
                else:
                    print(y_string + " duplicated!")
    return y 
                

    
def desc_2_tags(X_string, brand_string, model_string, frequent_terms, features_list, words_list):
    tokens_list = wordpunct_tokenize(X_string)    
    y = ["NA" for i in range(len(tokens_list))]
    # brand
    y = desc_2_tag(y, tokens_list, brand_string, "B")    
    # model
    y = desc_2_tag(y, tokens_list, model_string, "M")
    desc_features = [token_2_features(tokens_list, i,frequent_terms, features_list, words_list) for i in range(len(tokens_list))]
    return desc_features, y
def train_crf(model_id, X, y):
    if 'frequent_terms_eng.p' not in  os.listdir('models'):
        print('----- frequent_terms eng not found')    
        frequent_terms = frequent_terms_list(X, 10)
        pickle.dump( frequent_terms, open( "models/frequent_terms_eng.p", "wb" ) )
    else:
        print('---- frequent_terms eng found in models')
        frequent_terms = pickle.load( open( "models/frequent_terms_eng.p", "rb" ) ) 
    
    if 'words_list_eng.p' not in os.listdir('models'):
        words_list = read_words_list ('words_list_eng.csv')
        pickle.dump(words_list, open('models/words_list_eng.p', 'wb'))
    else:
        words_list = pickle.load( open( "models/words_list_eng.p", "rb" ) ) 
    
    params = {
    'c1': 10 ** (-3),   # coefficient for L1 penalty
    'c2': 10 ** (-5),  # coefficient for L2 penalty
    'max_iterations': 100,  
    }
    X_train = []
    y_train = []
    for i in range(len(X)):        
        desc_features, blm_features = desc_2_tags(X[i], y[i,0],y[i,1], frequent_terms, init_features_list, words_list)        
        X_train.append(desc_features)
        y_train.append(blm_features)
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.select(algorithm = 'lbfgs')
    trainer.set_params(params)         
    for xseq, yseq in zip(X_train, y_train):        
        trainer.append(xseq, yseq)    
    trainer.train(model_id)        
    return model_id

def tag_seq(crf_model_id, X_test):
    tagger = pycrfsuite.Tagger()
    tagger.open(crf_model_id)
    crf_features = init_features_list
    frequent_terms = pickle.load( open( "models/frequent_terms_eng.p", "rb" ) )
    word_list = pickle.load( open( "models/words_list_eng.p", "rb" ) ) 
    X_features = []
    y_true = []
    for i in range(len(X_test)):                        
        desc_features, _ = desc_2_tags(X_test[i], [], [], frequent_terms, crf_features,word_list)
        X_features.append(desc_features)        
    
    y_preds = []
    brand_beg_prob, model_beg_prob = [], []
    for xseq in X_features: 
        y_preds.append(tagger.tag(xseq))        
        brand_beg_prob.append([tagger.marginal('B-beg', i) for i in range(len(xseq))])                
        model_beg_prob.append([tagger.marginal('M-beg', i) for i in range(len(xseq))])
    
    prob_B, choice_B, label_B = [], [], []
    prob_M, choice_M, label_M = [], [], []
    for i in range(len(X_test)):        
        prob_Bi, choice_Bi, label_Bi  = prob_2_string(X_test[i], y_preds[i], brand_beg_prob[i], 'B')                
        prob_Mi, choice_Mi, label_Mi = prob_2_string (X_test[i], y_preds[i], model_beg_prob[i], 'M')        
        prob_B.append(prob_Bi);choice_B.append(choice_Bi), label_B.append(label_Bi)
        prob_M.append(prob_Mi); choice_M.append(choice_Mi), label_M.append(label_Mi)
    
    return prob_B, choice_B, label_B,prob_M, choice_M, label_M
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    