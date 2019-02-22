#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import svhn_exp
import cifar_exp
import fashion_exp
import mnist_exp
import statistic

import pandas as pd
import numpy as np

import keras
from keras.models import Model
from keras.layers import Dense,Lambda,Input
from keras import backend as K
from keras.optimizers import SGD,Adam
from sklearn.model_selection import train_test_split

from keras.utils import np_utils

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from tqdm import tqdm

from hyperopt import hp

def apfd(right,sort):
    length=np.sum(sort!=0)
    if length!=len(sort):
        sort[sort==0]=np.random.permutation(len(sort)-length)+length+1
    sum_all=np.sum(sort[[right!=1]])
    n=len(sort)
    m=pd.value_counts(right)[0]
    return 1-float(sum_all)/(n*m)+1./(2*n)

def gen_data():
    for dataset in ['svhn_exp','cifar_exp','fashion_exp','mnist_exp']:
        input,layers,test,train,pred_test,true_test,pred_test_prob = eval('{}.gen_data(use_adv=False)'.format(dataset))
        #正确是1错误是0
        is_right = (pred_test==true_test).astype('int')
        df = pd.DataFrame(pred_test_prob)
        df['label'] = is_right
        df.to_csv('./pred_test/{}.csv'.format(dataset))
        input,layers,test,train,pred_test,true_test,pred_test_prob = eval('{}.gen_data(use_adv=True)'.format(dataset))
        #正确是1错误是0
        is_right = (pred_test==true_test).astype('int')
        df = pd.DataFrame(pred_test_prob)
        df['label'] = is_right
        df.to_csv('./pred_test/{}_adv.csv'.format(dataset))

def model1():
    data = pd.read_csv('./pred_test/cifar_exp_adv.csv',index_col=0)
    feature = data[data.columns.difference(['label'])]
    feature=feature.applymap(lambda x:x**2)
    label = data['label']
    X_train,X_test,y_train,y_test=train_test_split(feature,label,test_size=0.2,random_state=0)
    batch_size = 100
    input_tensor_1 = Input(shape=(1,))
    #input_tensor_1 = K.constant(1,shape=(batch_size,1))
    input_tensor_2 = Input(shape=(10,))
    temp = Dense(10,activation='softmax',use_bias=False)(input_tensor_1)

    def f(inputs):
        x1,x2=inputs
        return K.reshape(K.sum(x1*x2,axis=1),(-1,1))
    #output_tensor = K.sum(temp*input_tensor_2,axis=1)
    output_tensor = Lambda(f)([temp,input_tensor_2])
    model = Model(inputs=[input_tensor_1,input_tensor_2],outputs=output_tensor)

    def my_loss(y_true, y_pred):
        loss = K.sum((K.ones_like(y_true)-y_true)*(y_pred),axis=1)+K.sum(y_true*y_pred,axis=1)
        return loss
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(lr=0.01, decay=1e-6,momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,loss=my_loss)
    model.fit([np.ones(len(X_train)).reshape(-1,1),X_train], y_train, batch_size=batch_size, nb_epoch=200,validation_data=([np.ones(len(X_test)).reshape(-1,1),X_test], y_test))


    sort_1 = model.predict([np.ones(len(X_test)).reshape(-1,1),X_test]).reshape(-1)
    sort_2 = np.sum(X_test,axis=1)
    print(apfd(y_test.values[np.argsort(sort_1)],np.arange(1,len(sort_1)+1)))
    print(apfd(y_test.values[np.argsort(sort_2)],np.arange(1,len(sort_2)+1)))

def model2():
    data = pd.read_csv('./pred_test/cifar_exp_adv.csv',index_col=0)
    feature = data[data.columns.difference(['label'])]
    feature=feature.applymap(lambda x:x**2)
    label = data['label']
    X_train,X_test,y_train,y_test=train_test_split(feature,label,test_size=0.2,random_state=0)

    y_train_oh = np_utils.to_categorical(y_train, 2)
    y_test_oh = np_utils.to_categorical(y_test, 2)

    input_tensor_1 = Input(shape=(10,))
    temp = Dense(2,activation='softmax',use_bias=True)(input_tensor_1)


    model = Model(inputs=input_tensor_1,outputs=temp)

    def my_loss(y_true, y_pred):
        loss = K.sum((K.ones_like(y_true)-y_true)*(y_pred),axis=1)+K.sum(y_true*y_pred,axis=1)
        return loss
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(lr=0.01,momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,loss='categorical_crossentropy')
    model.fit(X_train, y_train_oh, batch_size=100, nb_epoch=200,validation_data=(X_test, y_test_oh))


    sort_1 = model.predict(X_test)[:,1].reshape(-1)
    sort_2 = np.sum(X_test,axis=1)
    print(apfd(y_test.values[np.argsort(sort_1)],np.arange(1,len(sort_1)+1)))
    print(apfd(y_test.values[np.argsort(sort_2)],np.arange(1,len(sort_2)+1)))

def model3():
    data = pd.read_csv('./pred_test/cifar_exp_adv.csv',index_col=0)
    feature = data[data.columns.difference(['label'])]
    feature=feature.applymap(lambda x:x**2)
    label = data['label']
    X_train,X_test,y_train,y_test=train_test_split(feature,label,test_size=0.2,random_state=0)

    lr = LogisticRegression(max_iter=10000,n_jobs=-1,random_state=10000,C=1000,tol=0.0000001)
    lr.fit(X_train,y_train)
    sort_1 = lr.predict_proba(X_test)[:,1]
    sort_2 = np.sum(X_test,axis=1)
    print(apfd(y_test.values[np.argsort(sort_1)],np.arange(1,len(sort_1)+1)))
    print(apfd(y_test.values[np.argsort(sort_2)],np.arange(1,len(sort_2)+1)))


def model4():
    data = pd.read_csv('./pred_test/cifar_exp.csv',index_col=0)
    feature = data[data.columns.difference(['label'])]
    feature_2 = feature.applymap(lambda x:x**2)
    feature_1 = feature.copy()
    feature = pd.concat([feature_2,feature_1],axis=1)
    label = data['label']
    X_train,X_test,y_train,y_test=train_test_split(feature,label,test_size=0.2,random_state=0)

    y_train_oh = np_utils.to_categorical(y_train, 2)
    y_test_oh = np_utils.to_categorical(y_test, 2)

    input_tensor_1 = Input(shape=(20,))
    temp = Dense(2,activation='softmax',use_bias=True)(input_tensor_1)


    model = Model(inputs=input_tensor_1,outputs=temp)

    def my_loss(y_true, y_pred):
        loss = K.sum((K.ones_like(y_true)-y_true)*(y_pred),axis=1)+K.sum(y_true*y_pred,axis=1)
        return loss
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(lr=0.01,momentum=0.9, nesterov=True)
    adm = Adam(lr=0.001,decay=1e-6)
    model.compile(optimizer=adm,loss='categorical_crossentropy')
    model.fit(X_train, y_train_oh, batch_size=100, nb_epoch=1000,validation_data=(X_test, y_test_oh))


    sort_1 = model.predict(X_test)[:,1].reshape(-1)
    sort_2 = np.sum(X_test,axis=1)
    print(apfd(y_test.values[np.argsort(sort_1)],np.arange(1,len(sort_1)+1)))
    print(apfd(y_test.values[np.argsort(sort_2)],np.arange(1,len(sort_2)+1)))


def model5():
    data = pd.read_csv('./pred_test/cifar_exp_adv.csv',index_col=0)
    feature = data[data.columns.difference(['label'])]
    feature_2 = feature.applymap(lambda x:x**2)
    feature_1 = feature.copy()
    feature_3 = feature.applymap(lambda x:x**4)
    feature = pd.concat([feature_2,feature_1,feature_3],axis=1)
    label = data['label']
    X_train,X_test,y_train,y_test=train_test_split(feature,label,test_size=0.2,random_state=666)

    y_train_oh = np_utils.to_categorical(y_train, 2)
    y_test_oh = np_utils.to_categorical(y_test, 2)

    input_tensor_1 = Input(shape=(30,))

    temp = Dense(10,activation='relu',use_bias=True)(input_tensor_1)

    temp = Dense(2,activation='softmax',use_bias=True)(temp)


    model = Model(inputs=input_tensor_1,outputs=temp)

    def my_loss(y_true, y_pred):
        loss = K.sum((K.ones_like(y_true)-y_true)*(y_pred),axis=1)+K.sum(y_true*y_pred,axis=1)
        return loss
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(lr=0.01,momentum=0.9, nesterov=True)
    adm = Adam(lr=0.001,decay=1e-6)
    model.compile(optimizer=adm,loss='categorical_crossentropy')
    model.fit(X_train, y_train_oh, batch_size=100, nb_epoch=200,validation_data=(X_test, y_test_oh))


    sort_1 = model.predict(X_test)[:,1].reshape(-1)
    sort_2 = np.sum(X_test,axis=1)
    print(roc_auc_score(y_test,sort_1))
    print(roc_auc_score(y_test,sort_2))
    print(apfd(y_test.values[np.argsort(sort_1)],np.arange(1,len(sort_1)+1)))
    print(apfd(y_test.values[np.argsort(sort_2)],np.arange(1,len(sort_2)+1)))

def model6():

    from hyperopt import hp, fmin, rand, tpe, space_eval
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

    data = pd.read_csv('./pred_test/cifar_exp_adv.csv',index_col=0)
    feature = data[data.columns.difference(['label'])]
    feature=feature.applymap(lambda x:x**2)
    label = data['label']
    X_train,X_test,y_train,y_test=train_test_split(feature,label,test_size=0.2,random_state=0)

    def hyper_train_test(params):
        #print(np.array(list(params.values())))
        sort_1=np.dot(X_test.values,np.array(list(params.values()))).reshape(-1)
        return (apfd(y_test.values[np.argsort(sort_1)],np.arange(1,len(sort_1)+1)))

    def f(params):
        apfd_values = hyper_train_test(params)
        return {'loss': apfd_values, 'status': STATUS_OK}

    space = {'x1':hp.uniform('x1', 0, 1),'x2':hp.uniform('x2', 0, 1),'x3':hp.uniform('x3', 0, 1),'x4':hp.uniform('x4', 0, 1),'x5':hp.uniform('x5', 0, 1)\
    ,'x6':hp.uniform('x6', 0, 1),'x7':hp.uniform('x7', 0, 1),'x8':hp.uniform('x8', 0, 1),'x9':hp.uniform('x9', 0, 1),'x10':hp.uniform('x10', 0, 1)}
    trials = Trials()
    sort_2 = np.sum(X_test,axis=1)
    print(apfd(y_test.values[np.argsort(sort_2)],np.arange(1,len(sort_2)+1)))
    best = fmin(f, space, algo=tpe.suggest, max_evals=5000, trials=trials)
    print(best)
    print(space_eval(space, best))

def model7():
    data = pd.read_csv('./pred_test/cifar_exp_adv.csv',index_col=0)
    feature = data[data.columns.difference(['label'])]
    feature=feature.applymap(lambda x:x**2)
    label = data['label']
    X_train,X_test,y_train,y_test=train_test_split(feature,label,test_size=0.2,random_state=0)
    sort_train = np.sum(X_train,axis=1)
    value_train = apfd(y_train.values[np.argsort(sort_train)],np.arange(1,len(sort_train)+1))

    best_apfd = None
    best_value = None
    eps = 1/1000.
    temp = eps*X_train['1'].copy()
    for i in tqdm(range(1000)):
        X_train['1']=X_train['1']+temp
        sort_ = np.sum(X_train,axis=1)
        value_ = apfd(y_train.values[np.argsort(sort_)],np.arange(1,len(sort_)+1))
        if value_>value_train:
            print('value_')
            best_apfd = value_
            best_value = (i+1)/1000.

    sort_test = np.sum(X_test,axis=1)
    value_test = apfd(y_test.values[np.argsort(sort_test)],np.arange(1,len(sort_test)+1))

    X_test['0']=X_test['0']*(1+best_value)
    sort_test_ = np.sum(X_test,axis=1)
    value_test_ = apfd(y_test.values[np.argsort(sort_test_)],np.arange(1,len(sort_test_)+1))
    return value_test,value_test_,best_value

if __name__=='__main__':
    #value_test,value_test_,best_value = model7()
    #print(value_test,value_test_,best_value)
    model5()
