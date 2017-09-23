#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-
'''
Created on Jul 9, 2015

@author: HCH
'''
import pandas as pd
import numpy as np
from sklearn import cross_validation, linear_model, ensemble, metrics,svm
import BM_eng, BM_nor
import csv as csv
import re
import sys
import codecs
import json

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
SEED = 2015

def similar_score(actual_text, predicted_text):
    actual_break = re.split(' ', actual_text)
    predicted_break = re.split(' ', predicted_text)
    score = 0
    for actual_w in actual_break:
        if actual_w in predicted_break:
            score += 1
    return score

def split_data():
    data = pd.read_csv('data.csv', encoding = 'utf8')
    data['brand'] = data['norbrand']
    data['model'] = data['normodel']
    B_not_nor = 0
    M_not_nor = 0
    for idx in range(data.shape[0]):                   
        if str(data.loc[idx, 'norbrand']) == 'nan' and str(data.loc[idx, 'engbrand']) != 'nan':
            data.loc[idx, 'brand'] = data.loc[idx, 'engbrand']
            B_not_nor += 1
        if str(data.loc[idx, 'normodel']) == 'nan' and str(data.loc[idx, 'engmodel']) != 'nan':
            data.loc[idx, 'model'] = data.loc[idx, 'engmodel']
            M_not_nor += 1
    print('--- Num of brand from eng: %d' % B_not_nor)
    print('--- Num of model from eng: %d' % M_not_nor)
    print('-- done create final brand + model')        
    skf = cross_validation.ShuffleSplit(n = data.shape[0],  n_iter = 1, test_size = 0.3, random_state = SEED)
    for train_idx, test_idx in skf:
        train, test = data.ix[train_idx, :], data.ix[test_idx, :]
    print('-- done shuffle split')
    df = pd.DataFrame(train, columns= list(data.columns))
    df.to_csv('data/train.csv', index= False, encoding= 'utf8')
    df = pd.DataFrame(test, columns = list(data.columns))
    df.to_csv('data/test.csv', index= False, encoding= 'utf8')    
    print('-- done writing data')

def cross_prob_train(kf, X, y, y_selected, target = 'nor'):
    print('--- begin cross prob train')
    X_prob_B, X_prob_M = [], []
    X_choice_B, X_choice_M = [], []
    X_label_B, X_label_M = [], []
    label_B_1, label_M_1 = [], []
    X_str, y_str = [], []
    for train_idx, test_idx in kf:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test, y_test_selected = y[train_idx], y[test_idx], y_selected[test_idx]
        label_selected_B = y_test_selected[:, 0]; label_selected_M = y_test_selected[:, 1]   
        X_str.append(X_test);        
        if target == 'nor':
            crf_model = BM_nor.train_crf( 'models/crf_temp_nor', X_train, y_train)
            prob_B, choice_B, label_B, prob_M, choice_M, label_M = BM_nor.tag_seq(crf_model, X_test)             
            y_str.append(y_test[:, [1,2]])
        if target == 'eng':
            crf_model = BM_eng.train_crf('models/crf_temp_eng', X_train, y_train)
            prob_B, choice_B, label_B, prob_M, choice_M, label_M = BM_eng.tag_seq(crf_model, X_test)             
            y_str.append(y_test)
            
        X_prob_B.append(prob_B); X_prob_M.append(prob_M)
        X_choice_B.append(choice_B); X_choice_M.append(choice_M)
        X_label_B.append(label_B); X_label_M.append(label_M)
        label_B_1.append(label_selected_B), label_M_1.append(label_selected_M)
    print('--- done cross train, start train crf for whole training data')    
    X_prob_B, X_prob_M = np.hstack(X_prob_B), np.hstack(X_prob_M)
    X_choice_B, X_choice_M = np.hstack(X_choice_B), np.hstack(X_choice_M)
    X_label_B, X_label_M = np.hstack(X_label_B), np.hstack(X_label_M)
    label_B_1, label_M_1 = np.hstack(label_B_1), np.hstack(label_M_1)
    if target == 'nor':
        crf_model = BM_nor.train_crf('models/train_crf_nor', X, y)
    else:
        crf_model = BM_eng.train_crf('models/train_crf_eng', X,y)
    print('--- done train, start writing to file')
#     write to files
    X_str = np.hstack(X_str); y_str = np.vstack(y_str)     
    fi = csv.writer(open ('data/train_prob_text_' + target + '.csv', 'w',  encoding='utf8'), delimiter=',',  lineterminator='\n')
    fi.writerow(['text', 'true brand', 'predicted brand', 'choice brand', 'prob brand', 
                 'true model', 'predicted model', 'choice model', 'prob model' ])
    for idx in range(len(X_str)):
        if X_choice_B[idx] == 0 or X_choice_M[idx] == 0:
            fi.writerow([X_str[idx], y_str[idx, 0], X_label_B[idx], X_choice_B[idx], X_prob_B[idx],
                    y_str[idx, 1], X_label_M[idx], X_choice_M[idx], X_prob_M[idx]])
    print('--- done writing to file, done cross_prob_train')
    print('---')    
    return X_prob_B, X_prob_M, X_choice_B, X_choice_M, X_label_B, X_label_M, label_B_1, label_M_1, crf_model

def create_prob_Xy (X_prob_nor, label_nor, X_choice_nor, X_prob_eng, label_eng, X_choice_eng, label_1, target):
    print('--- start create_prob_Xy, start creating y label 0/1/2')
    y_prob = np.zeros(len(X_prob_nor))
    for idx in range(len(y_prob)):
        if target == 'brand':
            if str(label_1[idx]).lower() == str(label_eng[idx]).lower() and str(label_1[idx]).lower() == str(label_nor[idx]).lower():
                if X_prob_nor[idx] > X_prob_eng[idx]:
                    y_prob[idx] = 1
                else:
                    y_prob[idx] = 2
            elif str(label_1[idx]).lower() == str(label_eng[idx]).lower():
                y_prob[idx] = 2
            elif str(label_1[idx]).lower() == str(label_nor[idx]).lower():
                y_prob[idx] = 1
                   
        else: # target == 'model' 
            label_nor_scr = similar_score(str(label_1[idx]).lower(), str(label_nor[idx]).lower())
            label_eng_scr = similar_score(str(label_1[idx]).lower(), str(label_eng[idx]).lower())
            if label_nor_scr > label_eng_scr:
                y_prob[idx] = 1
            elif label_eng_scr > label_nor_scr:
                y_prob[idx] = 2
            elif label_eng_scr == label_nor_scr and label_nor_scr > 1:
                y_prob[idx] = 2
    print('--- done creating y, start hstack X')
    X_prob = np.hstack([X_prob_nor.reshape(-1,1), X_choice_nor.reshape(-1,1), X_prob_eng.reshape(-1,1), X_choice_eng.reshape(-1,1)])
    assert X_prob.shape[0] == len(y_prob), 'Dimensions not match X_prob and y_prob' 
    print('--- done create_prob_Xy')
    print('---')   
    return X_prob, y_prob
def create_prob_train():
    data = pd.read_csv('data/train.csv', encoding = 'utf8')
    X_nor = data.ix[:, 0].values
    X_eng = data.ix[:, 1].values
    print(data.shape)
    y_nor = data.loc[:, ['norlocation', 'norbrand', 'normodel']].values
    print(y_nor.shape)
    y_eng = data.loc[:, ['engbrand', 'engmodel']].values    
    y_selected = data.loc[:, ['brand', 'model']].values 
    n_folds = 10
    kf = cross_validation.KFold(n = data.shape[0], n_folds = n_folds, random_state = SEED)
    
    X_prob_B_eng, X_prob_M_eng, X_choice_B_eng, X_choice_M_eng, label_B_eng, label_M_eng, label_B_2, label_M_2, crf_eng = cross_prob_train(kf, X_eng, y_eng, y_selected, target= 'eng')
    X_prob_B_nor, X_prob_M_nor, X_choice_B_nor, X_choice_M_nor, label_B_nor, label_M_nor, label_B_1, label_M_1, crf_nor = cross_prob_train(kf, X_nor, y_nor, y_selected, target= 'nor')
        
    assert (label_B_1 == label_B_2).all , 'label_B_1 and label_B_2 not match'
    assert (label_M_1 == label_M_2).all, 'label_M_1 and label_M_2 not match'
#     create brand prob data
    B_nor_ndchoice = [idx for idx in range(len(X_choice_B_nor)) if X_choice_B_nor[idx] == 0]    
    X_prob_B, y_prob_B = create_prob_Xy (X_prob_B_nor[B_nor_ndchoice], label_B_nor[B_nor_ndchoice], X_choice_B_nor[B_nor_ndchoice],
                                         X_prob_B_eng[B_nor_ndchoice], label_B_eng[B_nor_ndchoice], X_choice_B_eng[B_nor_ndchoice], 
                                         label_B_1[B_nor_ndchoice], target = 'brand')
    df = pd.DataFrame(np.hstack([X_prob_B, y_prob_B.reshape(-1,1)]))
    df.to_csv('data/train_prob_brand.csv', header = False, index = False)
#     create model prob data
    M_nor_ndchoice = [idx for idx in range(len(X_choice_M_nor)) if X_choice_M_nor[idx] == 0]  
    X_prob_M, y_prob_M = create_prob_Xy (X_prob_M_nor[M_nor_ndchoice], label_M_nor[M_nor_ndchoice], X_choice_M_nor[M_nor_ndchoice],
                                         X_prob_M_eng[M_nor_ndchoice], label_M_eng[M_nor_ndchoice], X_choice_M_eng[M_nor_ndchoice], 
                                         label_M_1[M_nor_ndchoice], target = 'model')
    df = pd.DataFrame(np.hstack([X_prob_M, y_prob_M.reshape(-1,1)]))
    df.to_csv('data/train_prob_model.csv', header = False, index = False)
    return X_prob_B, y_prob_B, X_prob_M, y_prob_M, crf_nor, crf_eng                
def create_prob_test(crf_nor, crf_eng):
#     load data
    data = pd.read_csv('data/test.csv', encoding = 'utf8')
    X_nor = data.ix[:, 0].values
    X_eng = data.ix[:, 1].values        
    y_selected = data.loc[:, ['brand', 'model']].values   
    
#     predict Brand, Model
    X_prob_B_nor, X_choice_B_nor, label_B_nor, X_prob_M_nor, X_choice_M_nor, label_M_nor = BM_nor.tag_seq(crf_nor, X_nor)     
    X_prob_B_eng, X_choice_B_eng, label_B_eng, X_prob_M_eng,X_choice_M_eng, label_M_eng = BM_eng.tag_seq(crf_eng, X_eng)     
    assert len(X_choice_B_nor) == len(X_choice_M_nor), 'len of choice of B and M not match: %d, %d' % (len(X_choice_B_nor), len(X_choice_M_nor))
    
    B_nor_ndchoice = [idx for idx in range(len(X_choice_B_nor)) if X_choice_B_nor[idx] == 0]  
    X_prob_B, y_prob_B = create_prob_Xy (np.array(X_prob_B_nor)[B_nor_ndchoice], np.array(label_B_nor)[B_nor_ndchoice],
                                         np.array(X_choice_B_nor)[B_nor_ndchoice], 
                                         np.array(X_prob_B_eng)[B_nor_ndchoice], 
                                         np.array(label_B_eng)[B_nor_ndchoice], 
                                         np.array(X_choice_B_eng)[B_nor_ndchoice], y_selected[B_nor_ndchoice,0], target = 'brand')
    df = pd.DataFrame(np.hstack([X_prob_B, y_prob_B.reshape(-1,1)]))
    df.to_csv('data/test_prob_brand.csv', header = False, index = False)
#     create model prob data
    M_nor_ndchoice = [idx for idx in range(len(X_choice_M_nor)) if X_choice_M_nor[idx] == 0]
    X_prob_M, y_prob_M = create_prob_Xy (np.array(X_prob_M_nor)[M_nor_ndchoice], np.array(label_M_nor)[M_nor_ndchoice],
                                          np.array(X_choice_M_nor)[M_nor_ndchoice],
                                          np.array(X_prob_M_eng)[M_nor_ndchoice], np.array(label_M_eng)[M_nor_ndchoice],
                                          np.array(X_choice_M_eng)[M_nor_ndchoice], y_selected[M_nor_ndchoice,1], target = 'model')
    df = pd.DataFrame(np.hstack([X_prob_M, y_prob_M.reshape(-1,1)]))
    df.to_csv('data/test_prob_model.csv', header = False, index = False)
    
#     write test brand/model data
    test_B_file = csv.writer(open ('data/test_brand_text.csv', 'w',  encoding='utf8'), delimiter=',',  lineterminator='\n')
    test_B_file.writerow(['nortext', 'engtext', 'brand','norlabel', 'englabel','norprob', 'engprob', 'labelprob'])
    test_M_file = csv.writer(open ('data/test_model_text.csv', 'w',  encoding='utf8'), delimiter=',',  lineterminator='\n')
    test_M_file.writerow(['nortext', 'engtext', 'model','norlabel', 'englabel','norprob', 'engprob', 'labelprob'])
    B_idx, M_idx = 0, 0    
    for idx in range(len(X_choice_B_nor)):
        if X_choice_B_nor[idx] == 0:            
            test_B_file.writerow([X_nor[idx], X_eng[idx], y_selected[idx, 0], 
                              label_B_nor[idx], label_B_eng[idx], X_prob_B_nor[idx], X_prob_B_eng[idx], y_prob_B[B_idx]])
            B_idx += 1
        if X_choice_M_nor[idx] == 0:            
            test_M_file.writerow([X_nor[idx], X_eng[idx], y_selected[idx, 1], 
                              label_M_nor[idx], label_M_eng[idx], X_prob_M_nor[idx], X_prob_M_eng[idx], y_prob_M[M_idx]])
            M_idx += 1
    return X_prob_B, y_prob_B, X_prob_M, y_prob_M
def main():
    print('start split data')
    _ = split_data()
    print('done split data')
    X_B_train, y_B_train, X_M_train, y_M_train, crf_nor, crf_eng = create_prob_train()
#     B_train = pd.read_csv('data/train_prob_brand.csv')
    print('done create_prob_train')
    X_B_test, y_B_test, X_M_test, y_M_test = create_prob_test(crf_nor, crf_eng)
    print('done create_prob_test')
    
#     train prob_train data
    B_model = linear_model.LogisticRegression()
    M_model = linear_model.LogisticRegression()
    B_model.fit(X_B_train, y_B_train)
    print('done train prob Brand model')
    M_model.fit(X_M_train, y_M_train) 
    print('done train prob Model model')
         
#     evaluate prob_test data
    y_B_preds = B_model.predict(X_B_test)
    y_M_preds = M_model.predict(X_M_test)
    print('B model acc : %f' % metrics.accuracy_score(y_B_test, y_B_preds))
    print('M model acc : %f' % metrics.accuracy_score(y_M_test, y_M_preds))

def main2():
    B_train = pd.read_csv('data/train_prob_brand.csv', encoding = 'utf8', header = None).values
    M_train = pd.read_csv('data/train_prob_model.csv', encoding = 'utf8', header = None).values
    B_test = pd.read_csv('data/test_prob_brand.csv', encoding = 'utf8', header = None).values
    M_test = pd.read_csv('data/test_prob_model.csv', encoding = 'utf8', header = None).values
         
    X_B_train,y_B_train = B_train[:, : -1], B_train[:, -1]
    X_B_test,y_B_test = B_test[:, : -1], B_test[:, -1]
    X_M_train, y_M_train = M_train[:, : -1], M_train[:, -1]
    X_M_test, y_M_test = M_test[:, : -1], M_test[:, -1]
         
    #     train prob_train data
    B_model, B_model_id = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=0.0001, l1_ratio=0.15, n_iter= 100),'SGD log l2 a=0.0001 l1ratio=0.15'
    M_model, M_model_id = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=0.0001, l1_ratio=0.15, n_iter= 100),'SGD log l2 a=0.0001 l1ratio=0.15'
    B_model.fit(X_B_train, y_B_train)
    print('done train prob Brand model')
    M_model.fit(X_M_train, y_M_train) 
    print('done train prob Model model')
            
#     evaluate prob_test data
    y_B_preds = B_model.predict(X_B_test)
    y_M_preds = M_model.predict(X_M_test)
    print('B model acc : %f' % metrics.accuracy_score(y_B_test, y_B_preds))
    print('M model acc : %f' % metrics.accuracy_score(y_M_test, y_M_preds))
    B_test_file = pd.read_csv('output/test_B.csv')
    df = pd.DataFrame(np.hstack([B_test_file.values, y_B_preds.reshape(-1,1)]), columns = list(B_test_file.columns) + [B_model_id])
    df.to_csv('output/test_B.csv', index = False, encoding= 'utf8')
    M_test_file = pd.read_csv('output/test_M.csv')
    df = pd.DataFrame(np.hstack([M_test_file.values, y_M_preds.reshape(-1,1)]), columns = list(M_test_file.columns) + [M_model_id])
    df.to_csv('output/test_M.csv', index = False, encoding= 'utf8')    
def compute_threshold(tag_true, threshold_arr):        
    f_score_arr = []    
    for thres in threshold_arr:
        tag_preds = [0] * len(tag_true)
        for i in range(len(tag_preds)):
            if threshold_arr[i] < thres:
                tag_preds[i] = 0
            else:
                tag_preds[i] = 1                
        f_score_arr.append(metrics.f1_score(tag_true, tag_preds))
    max_val = np.amax(f_score_arr)
    max_index = 0
    for j,val in enumerate(f_score_arr):
        if val == max_val:
            max_index = j
            break;
    thres_res = threshold_arr[max_index]
    print(f_score_arr[: 30])
    return thres_res, max_val
def find_threshold(target):
    if target == 'B':
        train = pd.read_csv('data/train_prob_brand.csv', header = None)
        test = pd.read_csv('data/test_prob_brand.csv', header = None)
        data = pd.concat([train, test], ignore_index= True)
    elif target == 'M':
        train = pd.read_csv('data/train_prob_model.csv', header = None)
        test = pd.read_csv('data/test_prob_model.csv', header = None)
        data = pd.concat([train, test], ignore_index= True)
    print(data.shape)
    remove_idx = [idx for idx in range(data.shape[0]) if data.ix[idx, 4] == 2]
    data = data.drop(remove_idx, axis = 0)
    print(data.shape)
    threshold_arr = data.values[:, 0]
    true_tag = data.values[:, 4]
    thres_res, max_val = compute_threshold(true_tag, threshold_arr)    
    return thres_res, max_val
def train_full_models():
    #     train full model nor
    data = pd.read_csv('data.csv')
    data_nor = pd.concat([pd.read_csv('train_nor_2122.csv', header = None), pd.read_csv('test_nor_2122.csv', header = None)]
                         , ignore_index = True)
    X_nor = np.hstack([data.values[:,0], data_nor.values[:,0]])
    print(len(X_nor))
    y_nor = np.vstack([data.loc[:, ['norlocation', 'norbrand', 'normodel']].values, data_nor.ix[:,[1,2,3]].values])
    print(y_nor.shape)    
    crf_nor_id = BM_nor.train_crf( 'models/crf_nor', X_nor , y_nor)

#     train full model eng
    data = pd.read_csv('data.csv')
    data_eng = pd.read_csv('data_eng.csv', header = None)
    X_eng = np.hstack([data.values[:,1], data_eng.values[:,0]])
    print(len(X_eng))
    y_eng = np.vstack([data.loc[:, ['engbrand', 'engmodel']].values, data_eng.ix[:,[1,2]].values])
    print(y_eng.shape)
    crf_eng_id = BM_eng.train_crf( 'models/crf_eng', X_eng , y_eng)
def pick_B_M(prob_nor, choice_nor, label_nor, prob_eng, choice_eng, label_eng, target):
    B_nor_threshold = 0.063378
    M_nor_threshold = 0.008949
    B_eng_threshold = 0.0277101836238
    M_eng_threshold = 0.0596718382494
    if choice_nor == 1:
        return label_nor
    else:
        if choice_eng == 1:
            return label_eng
        else:
            if target == 'B':
                if prob_nor > B_nor_threshold:
                    return label_nor
                elif prob_eng > B_eng_threshold:
                    return label_eng
                else:
                    return ''
            elif target == 'M':
                if prob_nor > M_nor_threshold:
                    return label_nor
                elif prob_eng > M_eng_threshold:
                    return label_eng
                else:
                    return ''
    
if __name__ == '__main__':
#     train_full_models()
#     tag a sequence
    nor_text = sys.argv[1]
    eng_text = sys.argv[2]
    nor_text = nor_text.encode('utf-8', 'surrogateescape').decode('utf-8')
    eng_text = eng_text.encode('utf-8', 'surrogateescape').decode('utf-8')    
    prob_B_nor, choice_B_nor, label_B_nor,prob_M_nor, choice_M_nor, label_M_nor = BM_nor.tag_seq('BM_nor_eng/models/crf_nor', [nor_text])    
    prob_B_eng, choice_B_eng, label_B_eng,prob_M_eng, choice_M_eng, label_M_eng = BM_eng.tag_seq('BM_nor_eng/models/crf_eng', [eng_text])    
    B_res = pick_B_M(prob_B_nor[0], choice_B_nor[0], label_B_nor[0], prob_B_eng[0],choice_B_eng[0], label_B_eng[0], target = 'B')
    M_res = pick_B_M(prob_M_nor[0], choice_M_nor[0], label_M_nor[0], prob_M_eng[0], choice_M_eng[0], label_M_eng[0], target = 'M')
    print (json.dumps({"Brand": B_res, "Model": M_res}, ensure_ascii=False, separators=(',', ': ')))     






