
# coding: utf-8

# In[1]:

# import stuff
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from nltk import FreqDist
import numpy as np


# In[2]:

#load dataset
x_train_all = np.load('imdb/x_train.npy')
x_test_all = np.load('imdb/x_test.npy')
y_train = np.load('imdb/y_train.npy')
y_test = np.load('imdb/y_test.npy')


# In[3]:

# function to calc freqdist

def freqdist(x_all):
    for row in x_all:
        row.sort()
        
    fdist = [FreqDist(row) for row in x_all]
    
    return fdist


# In[4]:

#function to obtain top-K

def topk(k,fdistx):
    new_fdx = []
    
    for freqdist in fdistx:
        new_fdx.append([freqdist[i] for i in range(1,k+1)])  
    
    return np.array(new_fdx)


# In[5]:

def GaussianNaiveBayes():
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred = gnb.predict(x_test)
    #print((y_pred != y_test).sum())
    accuracy = accuracy_score(y_test,y_pred) * 100
    precision = precision_score(y_test,y_pred) * 100
    recall = recall_score(y_test,y_pred) * 100
    
    return accuracy,precision,recall


# In[ ]:

print('Calculating freqdist of x_train & x_test...',end='')
fd_xtrain = freqdist(x_train_all)
fd_xtest = freqdist(x_test_all)
print('done.')

for k in [100,1000,10000]: 

    print('~' * 10, ' K = %d '% k, '~' * 10)
    
    print('Obtaining frequency of top-%d words in x_train...' % k, end='')
    x_train = topk(k,fd_xtrain)
    print('done.')
    
    print('Obtaining frequency of top-%d words in x_test...' % k,end='')    
    x_test = topk(k,fd_xtest)
    print('done.')
    
    print('Training gnb model...',end='')
    accuracy,precision,recall = GaussianNaiveBayes()
    print('done.')
    print('Accuracy = %.3f, Precision = %.3f, Recall = %.3f\n' % (accuracy,precision,recall))
    
input('Press any key to exit.')

