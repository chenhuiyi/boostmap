#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
＃BoostMap training algorithm


@author: Huiyi Chen, Han Xiao
"""

#pip install editdistance
import editdistance
import random
from numpy import *

#load data and construct data objects
'''data is of length 400000, split it into chunks of length 40 with step 10'''
data = open('DNAdata').read()

DATA_LENGTH = 400000

OBJECTS = []
try:
    for i in range(DATA_LENGTH // 10):
        object = data[i*10 : (i+4)*10]
        OBJECTS.append(object)
except:
    pass
        

'''generate reference objects and training sets both of length 1000'''
SET_LENGTH = 1000
random.shuffle(OBJECTS)
C = OBJECTS[0:SET_LENGTH] #reference sets
L = OBJECTS[SET_LENGTH : SET_LENGTH*2] #traning sets


'''generate distance matrix'''
CC = zeros((SET_LENGTH, SET_LENGTH))
LC = zeros((SET_LENGTH, SET_LENGTH))
LL = zeros((SET_LENGTH, SET_LENGTH))

for i in range(SET_LENGTH):
    for j in range(SET_LENGTH):
        if i == j:
            CC[i][j] = float("inf")
            LC[i][j] = float("inf")
            LL[i][j] = float("inf")
        else:
            CC[i][j] = editdistance.eval(C[i], C[j])
            LC[i][j] = editdistance.eval(L[i], C[j])
            LL[i][j] = editdistance.eval(L[i], L[j])
        

#some formula functions
'''classification function'''
def P(X, A, B):
    Dxa = editdistance.eval(X, A)
    Dxb = editdistance.eval(X, B)
    if Dxa < Dxb:
        return 1
    elif Dxa > Dxb:
        return -1
    else:
        return 0
    
'''generate training sets O'''
BETA = 30  #size of training triples
O = []
while len(O) != BETA:
    x = random.randint(0, SET_LENGTH-1)
    a = argmin(LL[x])
    b = random.randint(0, SET_LENGTH-1)
    X = L[x]
    A = L[a]
    while LL[x][b] == LL[x][a]:
        b = random.randint(0, SET_LENGTH-1)
    B = L[b]
    o = [(X, A, B), P(X, A, B)]
    O.append(o)
    

'''use boostmap to train the data'''
weights = [ones(BETA)/BETA]
hList = []
aList = []
zList = []

'''the formula for embedding'''
def F(X, p):
    #if type(p) == int:
    i = L.index(X)
    return LC[i][p]
    '''else:
        i = L.index(X)
        d = (LC[i][p[0]] ** 2 + CC[p[0]][p[1]] ** 2 - LC[i][p[1]] **2) / (2*CC[p[0]][p[1]])
        return d'''


'''for each training round, select gamma weak classifier'''
gamma = 10
pList = [random.randint(0, SET_LENGTH) for i in range(gamma//2)]

'''find h and alpha'''
def Z(j, p, alpha):
    z = 0
    for i in range(BETA):
        z += weights[j][i] * exp(-alpha * O[i][1] * H(p, O[i][0][0], O[i][0][1], O[i][0][2]))
    return z

'''the classifier formula'''
def H(p, X, A, B):
    return abs(F(X, p) - F(B, p)) - abs(F(X, p) -F(A, p))

'''
for p in pList:
    errors = array([o[1] != sign(H(p, o[0][0], o[0][1], o[0][2])) for o in O])
    e = (errors*weights[j]).sum()
    alpha = 0.5 * log((1-e)/e)
    aList.append(alpha)
    hList.append(p)
    
for x in xList:
    errors = array([o[1] != sign(H(x, o[0][0], o[0][1], o[0][2])) for o in O])
    e = (errors*weights[j]).sum()
    alpha = 0.5 * log((1-e)/e)
    aList.append(alpha)
    hList.append(x)
'''
for j in range(BETA):
    p = random.randint(0, SET_LENGTH)
    errors = array([o[1] != sign(H(p, o[0][0], o[0][1], o[0][2])) for o in O])
    e = (errors*weights[j]).sum()
    alpha = 0.5 * log((1-e)/e)
    aList.append(alpha)
    hList.append(p)
    w = zeros(BETA)
    for i in range(BETA):
        w[i] = weights[j][i] * exp(-alpha * O[i][1] * sign(H(p, O[i][0][0], O[i][0][1], O[i][0][2])))
        w = w/w.sum()
        weights.append(w) 
        

aList = aList[2:]
hList = hList[2:]

'''define embedding formular'''
embedding = []
for o in OBJECTS:
    emb = 0
    for index in range(len(aList)):
        emb += aList[index] * editdistance.eval(o, C[hList[index]])
    embedding.append(emb)
    
    
'''embed query'''
def embed(Q):
    emb = 0
    for index in range(len(aList)):
        emb += aList[index] * editdistance.eval(Q, C[hList[index]])
    return emb

'''filter step'''
def filter(Fq):
    rank = embedding
    rank = [abs(i - Fq) for i in rank]
    rank = asarray(rank)
    pList = rank.argsort()[:5000]
    return pList

'''refine step'''
def refine(pList, Q):
    dList = zeros(len(pList))
    for i, o in enumerate(pList):
        d = editdistance.eval(Q, OBJECTS[o])
        dList[i] = d
    NN = dList.argsort()[0]
    return OBJECTS[pList[NN]]
    

def findNNemb(Q):
    NN = refine(filter(embed(Q)), Q)
    return (NN, editdistance.eval(Q, NN))
   

'''find NN using brute force'''
def NN(Q):
    distance = float('inf')
    NN = ''
    for o in OBJECTS:
        if editdistance.eval(Q, o) < distance:
            distance = editdistance.eval(Q, o)
            NN = o
    return (NN, distance)
    
    
    



     


    
    








