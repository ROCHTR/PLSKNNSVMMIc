# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 21:04:03 2018

@author: RC-X550Z
"""

import numpy
import csv
import math
import operator
import time
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC

random_index = numpy.array([37,24,39,15,38,44,48,29,49,9,50,35,18,16,22,12,46,60,10,27,26,42,21,1,47,41,51,45,52,56,7,17,2,43,2,14,58,59,55,11,32,6,57,30,23,53,19,31,25,33,36,4,8,54,5,0,61,13,20,40,34,3])

def my_NIPALS_train(X_train,Y_train,N_Components):
    U = []
    T = []
    P = []
    Q = []
    W = []
    
    train = X_train
    kelas = Y_train
    
    n = N_Components;
    
    E = train
    Emean = numpy.mean(E,axis=0)
    E -= Emean
    F = numpy.transpose(kelas)
    F = F.astype(float)
    Fmean = numpy.mean(F,axis=0)
    F -= Fmean
    F = numpy.reshape(F,(len(F),1))
    told = numpy.ones((len(E),1))*100
    zain = 0
    for i  in range(0,n):
        u = F[:]
        while True:
            w = numpy.dot(numpy.transpose(E),u)/numpy.linalg.norm(numpy.dot(numpy.transpose(E),u))
            t = numpy.dot(E,w)
            t = numpy.nan_to_num(t, 0)
            if(numpy.linalg.norm(told-t)<1e-5):
                break
            told = t
            told = numpy.nan_to_num(told, 0)
        #zain+=1
        p=numpy.dot(numpy.transpose(E),t)/numpy.linalg.norm(numpy.dot(numpy.transpose(t),t))
        #pnew = p/numpy.linalg.norm(p)
        #tnew = numpy.dot(t,numpy.linalg.norm(p))
        
        if(numpy.linalg.norm(t)<1e-5):
            t = numpy.zeros((len(E),1))
            p = numpy.zeros((len(p),1))
            w = numpy.zeros((len(w),1))
        
        T.append(t)
        P.append(p)
        W.append(w)
        E = E - numpy.dot(t,numpy.transpose(p))
    
    #print(numpy.linalg.norm(t))
    W = numpy.array(numpy.transpose(W))
    W = numpy.reshape(W,(len(W[0]),n))
    T = numpy.array(numpy.transpose(T))
    T = numpy.reshape(T,(len(T[0]),n))
    P = numpy.array(numpy.transpose(P))
    P = numpy.reshape(P,(len(P[0]),n))
    P = numpy.nan_to_num(P, 0)
    W = numpy.nan_to_num(W, 0)
    T = numpy.nan_to_num(T, 0)
    
    return W

def my_NIPALS_Fit_train(X_train,W):
    X_train = numpy.array(X_train)
    Xmean = numpy.mean(X_train,axis=0)
    X_train -= Xmean
    T_train = numpy.dot(X_train,W)
    
    return T_train

def my_NIPALS_test(X_test,W):
    X_test = numpy.array(X_test)
    Xmean = numpy.mean(X_test,axis=0)
    X_test -= Xmean
    T_test = numpy.dot(X_test,W)
    
    return T_test

from sklearn.preprocessing import normalize
def my_HKNNSVM(X_train, X_test, Y_train, K_Neighbors, Kernel_SVM):
    train = X_train
    train = normalize(train)
    test = X_test
    test = normalize(test)
    kelas = Y_train
    k = K_Neighbors
    kernel = Kernel_SVM
    hasilkelas = []
    Y_pred = []
    
    for z in range(0,len(test)):
        distance = []
        
        train = numpy.array(train)
        test = numpy.array(test)
        index_train = numpy.arange(len(train))
        index_train = index_train.tolist()
        length = len(train)
        for i in range(0,length):
            distance.append((math.sqrt(sum([(a - b)**2 for a, b in zip(train[i], test[z])])),kelas[i],tuple(train[i]),kelas[i]))
    
        distance.sort(key=operator.itemgetter(0))
        
        neighbor = []
        ttg = []
        kelasttg = []
        jarak = []
        for j in range(k):
            neighbor.append(distance[j])
            ttg.append(neighbor[j][2])
            kelasttg.append(neighbor[j][1])
            jarak.append(distance[0])
        
        ttg = list(ttg)
        classVotes = {}
        for a in range(len(neighbor)):
        	response = neighbor[a][1]
        	if response in classVotes:
        		classVotes[response] += 1
        	else:
        		classVotes[response] = 1
        #sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    
        items = []
        items = list(classVotes.items())
        #print(classVotes)
        '''svm'''
        if len(items) > 1:
            clf = OneVsOneClassifier(SVC(kernel=kernel))
            clf.fit(list(ttg),list(kelasttg))
            ley = [list(test[z])]
            hasilkelas = clf.predict(ley)
            #print(hasilkelas)
        else:
            hasilkelas = max(classVotes.items(), key=operator.itemgetter(1))[0]
            #print(hasilkelas)
            hasilkelas = numpy.reshape(hasilkelas,(1,))
            hasilkelas = numpy.array(hasilkelas)
        '''svm'''
        
        Y_pred.append(hasilkelas)
        
    return Y_pred

def LOAD_Colon():
    data_train = []
    data_all = []
    data_all_kelas = []
    data_all_attr = []
    with open('colonTumor.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            data_train.append(row)
    
    data_train = list(filter(None, data_train))
    
    data_train = numpy.array(data_train)
    data_all = data_train
    data_all = data_all[random_index]
    len_data_all = len(data_all[0])
    
    for i in range(0,len(data_all)):
        data_all_kelas.append(data_all[i][len_data_all-1])
        data_all_attr.append([float(x) for x in data_all[i][0:len_data_all-1]])
    
    data_all_kelas = numpy.array(data_all_kelas)
    data_all_attr = numpy.array(data_all_attr)
    
    
    data_all_kelas = list(map(int,data_all_kelas))
    data_all_kelas = numpy.array(data_all_kelas)
    
    return data_all_kelas, data_all_attr
#from sklearn.cross_validation import train_test_split
#from sklearn import metrics

#x_train, x_test, y_train, y_test = train_test_split(data_all_attr,data_all_kelas,random_state=5)

#Ypred = my_HKNNSVM(x_train, x_test, y_train, 10, 'linear')

def calculate_confusion(label, prediction):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
 
    for i in range(0, len(label)):
        if prediction[i] == 1:
            if prediction[i] == label[i]:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if prediction[i] == label[i]:
                true_negatives += 1
            else:
                false_negatives += 1
    
    return true_positives, false_positives, true_negatives, false_negatives

def calculate_F1(true_positives, false_positives, true_negatives, false_negatives): 
    # a ratio of correctly predicted observation to the total observations
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
 
    # precision is "how useful the search results are"
    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)
        
    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)
        
    if recall == 0 or precision == 0:
        f1_score = 0
    else:
        f1_score = 2 / ((1 / precision) + (1 / recall))
    # recall is "how complete the results are"
 
    return accuracy, precision, recall, f1_score

from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import euclidean_distances
kelas, attr = LOAD_Colon()
n_compon = []
file = open('Colon_Revisi_Sidang.txt', 'w')
attr_key = numpy.array([2,3,4,5,6,7,8,9,10])
pjg_key = len(attr_key)
ksplits = 5
N_comp = 50
for ikey in range(0,pjg_key):
    
    '''###### ISIAN #####'''
    k_neighbor_N = attr_key[ikey]
    kernel = euclidean_distances
    '''###### ISIAN #####'''
    
    
    #from sklearn.metrics import f1_score
    total_score = 0
    total_acc = 0
    total_tp = 0
    total_tn = 0
    total_time = 0
    Kfo = KFold(n_splits=ksplits, shuffle=False)
    for train_index, test_index in Kfo.split(attr):
        #print("TRAIN :", train_index, "TEST :", test_index)
        x_train = attr[train_index]
        x_test = attr[test_index]
        y_train = kelas[train_index]
        y_test = kelas[test_index]
        #T_train, P_train = my_NIPALS_train(x_train,y_train,N_comp)
        W_train = my_NIPALS_train(x_train,y_train,N_comp)
        T_train = my_NIPALS_Fit_train(x_train, W_train)
        T_test = my_NIPALS_test(x_test,W_train)
        n_compon.append(T_train[0])
        start = time.time()
        Ypred = my_HKNNSVM(T_train, T_test, y_train, k_neighbor_N, kernel)
        end = time.time()
        time_score = end - start
        Ypred = numpy.array(Ypred)
         #score = f1_score(y_test, Ypred, average='binary')
        tp, fp, tn, fn = calculate_confusion(y_test, Ypred)
        accuracy, precision, recall, score_f1 = calculate_F1(tp, fp, tn, fn)
        '''
        print("F1 : ", score_f1)
        print("Accuracy : ", accuracy)
        print("True Positive : ",tp)
        print("True Negative : ",tn)
        print("Running Time : ",time_score)
        '''
        total_score = total_score + score_f1
        total_acc = total_acc + accuracy
        total_tp = total_tp + tp
        total_tn = total_tn + tn
        total_time = total_time + time_score
        print("___________________________________________________")
    '''
    print("N Components PLS = ",N_comp)
    print("N Neighbor KNN = ",k_neighbor_N)
    print("Kernel SVM = ",kernel)
    AVG_Score = total_score / ksplits
    print("Average F1 = ", AVG_Score)
    AVG_Acc = total_acc / ksplits
    print("Average Accuracy = ", AVG_Acc)
    AVG_TP = total_tp / ksplits
    print("Average True Positive = ", AVG_TP)
    AVG_TN = total_tn / ksplits
    print("Average True Negative = ", AVG_TN)
    AVG_Time = total_time / ksplits
    print("Average Time Score = ", AVG_Time)
    print("######################################################")
    '''
    AVG_Score = total_score / ksplits
    AVG_Acc = total_acc / ksplits
    AVG_TP = total_tp / ksplits
    AVG_TN = total_tn / ksplits
    AVG_Time = total_time / ksplits
    print("N Components PLS, N Neighbor KNN, Kernel SVM, Average F1, Average Accuracy, Average True Positive, Average True Negative, Average Time Score")
    print(N_comp,",", k_neighbor_N,",", kernel,",", AVG_Score,",", AVG_Acc,",", AVG_TP,",", AVG_TN,",", AVG_Time)
    print("######################################################")
    file.write(str(N_comp) + " , " + str(k_neighbor_N) + " , " + str(kernel) + " , " + str(AVG_Score) + " , " + str(AVG_Acc) + " , " + str(AVG_TP) + " , " + str(AVG_TN) + " , " + str(AVG_Time) + '\n')
    
file.close()