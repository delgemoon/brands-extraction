#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:33:54 2015

@author: quetran
"""

import pycrfsuite
import crfTrain


import pickle
import sys
import codecs
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
def tagText(text):
    X_string_array = [text]
    tagger = pycrfsuite.Tagger()
    tagger.open('models/snapsale_crf_7')
    crf_features = crfTrain.init_features_list
    frequent_terms = pickle.load( open( "models/frequent_terms.p", "rb" ) )
    X_features = []
    
    for i in range(len(X_string_array)):                        
#        desc_features, _ = crfTrain.desc_2_feature(X_string_array[i], '', frequent_terms, crf_features)       
#        X_features.append(desc_features)
        desc_features, _ = crfTrain.desc_2_tags(X_string_array[i], '', '', '', frequent_terms, crf_features)
        X_features.append(desc_features)
    
    y_preds = []
    brand_beg_prob = []
    brand_in_prob = []
    location_beg_prob = []
    location_in_prob = []
    model_beg_prob = []
    model_in_prob = []
    brand_string_preds = []
    location_string_preds = []
    model_string_preds = []
    brand_conf = []
    location_conf = []
    model_conf = []    
    
    
    for xseq in X_features: 
        y_preds.append(tagger.tag(xseq))
        brand_beg_prob.append([tagger.marginal('B-beg', i) for i in range(len(xseq))])
        brand_in_prob.append([tagger.marginal('B-in', i) for i in range(len(xseq))])
        location_beg_prob.append([tagger.marginal('L-beg', i) for i in range(len(xseq))])
        location_in_prob.append([tagger.marginal('L-in', i) for i in range(len(xseq))])
        model_beg_prob.append([tagger.marginal('M-beg', i) for i in range(len(xseq))])
        model_in_prob.append([tagger.marginal('M-in', i) for i in range(len(xseq))])

    for i in range(len(X_string_array)):
        brand_pred_string, bc  = crfTrain.prob_2_string(X_string_array[i], y_preds[i], brand_beg_prob[i], brand_in_prob[i], "B")
        location_pred_string, lc = crfTrain.prob_2_string (X_string_array[i], y_preds[i], location_beg_prob[i], location_in_prob[i], "L")
        model_pred_string, mc = crfTrain.prob_2_string (X_string_array[i], y_preds[i], model_beg_prob[i], model_in_prob[i], "M")
        
        brand_string_preds.append(brand_pred_string)
        location_string_preds.append(location_pred_string)
        model_string_preds.append(model_pred_string)
        
        brand_conf.append(bc)
        location_conf.append(lc)
        model_conf.append(mc)

    
    return location_string_preds[0], location_conf[0], brand_string_preds[0], brand_conf[0], model_string_preds[0], model_conf[0]

if __name__ == '__main__':
    temp = sys.argv[1]
    temp = temp.encode('utf-8', 'surrogateescape').decode('utf-8')            
    l, lc, b, bc, m, mc = tagText(temp)
#     l, lc, b, bc, m, mc = tagText("Ralph Lauren female, super slim fit str. 8, lysblå.")
#    print("Location: %s\n" % l)
#    print("LocationConf: %f\n" % lc)
#    print("Brand: %s\n" % b)
#    print("BrandConf: %f\n" % bc)
#    print("Model: %s\n" % m)
#    print("ModelConf: %f\n"% mc)
#     s = "Ralph Lauren female, super slim fit str. 8, lysblå"
#     print (s)    
#    print("Ralph Lauren female, super slim fit str. 8, lysblå.")
    print("{\"Location\":\"%s\",\"LocationConf\":%f,\"Brand\":\"%s\",\"BrandConf\":%f,\"Model\":\"%s\",\"ModelConf\": %f}" % (l, lc, b, bc, m, mc))
