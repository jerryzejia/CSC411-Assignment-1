import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import scipy.misc
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from PIL import Image
import shutil
import copy
from numpy import *
from pylab import *

actor = ['Alec Baldwin']
Label = {'Alec Baldwin': [1,0,0,0,0,0]}

labels = []

def find_file_num(actor):
    file_num = 0
    for a in actor:
        dir = a+"/training/"
        for file in os.listdir(dir):
            file_num +=1
    return file_num

def set_creation_limited(actor, Label,file_num):
    labels =[]
    classfier = np.zeros((file_num*len(actor), 1025))
    i = 0
    for a in actor:
        j = 0
        dir = a+"/training/"
        for file in os.listdir(dir):
            img = imread(dir+file).astype(float)
            img = img/255
            img = img.reshape((1,1024))
            img = np.hstack(([[1]], img))
            classfier[i] = img
            labels.append((Label[a]))
            i+=1
            j+=1
            if j == file_num:
                break
    y = np.zeros((len(labels), len(labels[0])))
    for i in range(len(labels)):
        y[i] = labels[i]
    classfier = np.transpose(classfier)
    theta = np.zeros((1025, len(labels[0])))

    return classfier, y, theta



def f(x, y, theta):
    theta = np.transpose(theta)
    return sum((y.T - dot(theta, x)) ** 2)


def df(x, y, theta, norm):
    theta = np.transpose(theta)
    inner = dot(theta,x)-y.T
    return (2*(dot(x, inner.T)))


def grad_descent(f, df, x, y, init_t, alpha, max):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = max
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        gradient = df(x, y, t,t.shape[0]).reshape(1025,1)
        t -= alpha*gradient
        if iter % 10000 == 0:
            print("Iter %i: cost = %.2f" % (iter, f(x, y, t,t.shape[0])))
        iter += 1
    print("Iter %i: cost = %.2f" % (iter, f(x, y, t, t.shape[0])))
    return t


# x,y,theta = set_creation_limited(actor, Label,1)
# for i in theta:
#     for j in i:
#         j = random()
# new_theta = theta.copy()
# h = 0.0001
# dif = 0
# for k in range(5):
#     i = np.random.choice(len(theta))
#     j = np.random.choice(len(theta[0]))
#     new_theta[i][j] = theta[i][j] + h
#     dif += abs((f(x,y,new_theta+theta)-f(x,y,theta))/h-df(x,y,theta,1025)[i][j])
# print(str(dif/1025))










