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




def find_file_num(actor, add):
    file_num = 0
    for a in actor:
        dir = a+add
        for file in os.listdir(dir):
            file_num +=1
    return file_num

def set_creation(actor, Label, add):
    labels = []
    file_num = find_file_num(actor,add)
    classfier = np.zeros((file_num, 1025))
    i = 0
    for a in actor:
        dir = a+add
        for file in os.listdir(dir):
            img = imread(dir+file).astype(float)
            img = img/255
            img = img.reshape((1,1024))
            img = np.hstack(([[1]], img))
            classfier[i] = img
            labels.append(Label[a])
            i+=1
    y = np.zeros((len(labels), 1))
    for i in range(len(labels)):
        y[i] = labels[i]
    classfier = np.transpose(classfier)
    theta = np.zeros((1025, 1))
    return classfier, y, theta


def f(x, y, theta, norm):
    y = np.transpose(y)
    theta = np.transpose(theta)
    return sum((y - dot(theta, x)) ** 2)/(2*norm)


def df(x, y, theta, norm):
    y = np.transpose(y)
    theta = np.transpose(theta)
    return -2*sum((y-dot(theta, x))*x, 1)/norm


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


def test(theta,Label,actor,add):
    cor = 0
    cnt = 0
    for a in actor:
        i = 0
        dir = a + add
        for file in os.listdir(dir):
            img = imread(dir+file).astype(float)
            img = img/255
            img = img.reshape((1,1024))
            img = np.hstack(([[1]], img))
            prediction = np.dot(img, theta)[0]
            print("%s %i|pred: %.2f, ans: %.2f" % (a, i, prediction, Label[a]))
            ans = Label[a]
            if abs(Label[a] - prediction) < 1:
                cor += 1
            cnt+=1
    print 'score: '+ str(float(cor)/float(cnt))



# x,y,theta = set_creation(actor, Label)
# theta_p3 = grad_descent(f,df,x, y, theta, 0.001, 20000)
#
#
# test(theta_p3,Label,actor)
# plt.imshow(theta_p3[1:].reshape((32,32)))
# plt.show()







