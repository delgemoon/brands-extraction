'''
Created on Jul 21, 2015

@author: HCH
'''
import sys
import codecs
import json

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
def tag_seq(crf_model_id, X_test):
    tagger = pycrfsuite.Tagger()
    tagger.open(crf_model_id)
    X_string_array = X_test
    crf_features = init_features_list
    frequent_terms = pickle.load( open( "models/frequent_terms_nor.p", "rb" ) )
    word_form_dict = word_form_dict_create('models/dictionary.csv')
    X_features = []
    
    for i in range(len(X_string_array)):                                
        desc_features, _ = desc_2_tags(X_string_array[i], '', '', '', frequent_terms, crf_features, word_form_dict)
        X_features.append(desc_features)
    
    y_preds = []
    brand_beg_prob = []        
    model_beg_prob= []            
    for xseq in X_features: 
        y_preds.append(tagger.tag(xseq))
        brand_beg_prob.append([tagger.marginal('B-beg', i) for i in range(len(xseq))])        
        model_beg_prob.append([tagger.marginal('M-beg', i) for i in range(len(xseq))])        

    prob_B, choice_B, label_B = [], [], []
    prob_M, choice_M, label_M = [], [], []    
    for i in range(len(X_string_array)):
        prob_Bi, choice_Bi, label_Bi  = prob_2_string(X_string_array[i], y_preds[i],
                                                                        brand_beg_prob[i], 'B')
        
        prob_Mi, choice_Mi, label_Mi = prob_2_string (X_string_array[i], y_preds[i], 
                                                                       model_beg_prob[i],  'M')
        
        prob_B.append(prob_Bi);choice_B.append(choice_Bi), label_B.append(label_Bi)
        prob_M.append(prob_Mi); choice_M.append(choice_Mi), label_M.append(label_Mi)                    
    return prob_B, choice_B, label_B,prob_M, choice_M, label_M
def pick_B_M(prob_nor, choice_nor, label_nor, choice_eng, label_eng, target):
    B_threshold = 0.063378
    M_threshold = 0.008949
    if choice_nor == 1:
        return label_nor
    else:
        if choice_eng == 1:
            return label_eng
        else:
            if target == 'B':
                if prob_nor > B_threshold:
                    return label_nor
                else:
                    return 'null'
            elif target == 'M':
                if prob_nor > M_threshold:
                    return label_nor
                else:
                    return 'null'
if __name__ == '__main__':
    nor_text = sys.argv[1]
    eng_text = sys.argv[2]
    nor_text = nor_text.encode('utf-8', 'surrogateescape').decode('utf-8')
    eng_text = eng_text.encode('utf-8', 'surrogateescape').decode('utf-8')    
    prob_B_nor, choice_B_nor, label_B_nor,prob_M_nor, choice_M_nor, label_M_nor = tag_seq_nor('models/crf_nor', [nor_text])    
    prob_B_eng, choice_B_eng, label_B_eng,prob_M_eng, choice_M_eng, label_M_eng = tag_seq_eng('models/crf_eng', [eng_text])    
    B_res = pick_B_M(prob_B_nor[0], choice_B_nor[0], label_B_nor[0], choice_B_eng[0], label_B_eng[0], target = 'B')
    M_res = pick_B_M(prob_M_nor[0], choice_M_nor[0], label_M_nor[0], choice_M_eng[0], label_M_eng[0], target = 'M')
    print (json.dumps({"Brand": B_res, "Model": M_res}, ensure_ascii=False, separators=(',', ': '))) 
    