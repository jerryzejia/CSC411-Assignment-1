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


test_actor = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
Label  = {'Alec Baldwin': [1,0,0,0,0,0], 'Steve Carell':[0,1,0,0,0,0], 'Bill Hader':[0,0,1,0,0,0], 'Lorraine Bracco':[0,0,0,1,0,0], 'Peri Gilpin':[0,0,0,0,1,0], 'Angie Harmon':[0,0,0,0,0,1]}



def f(x, y, theta):
    theta = np.transpose(theta)
    return sum((y.T - dot(theta, x)) ** 2)


def df(x, y, theta, norm):
    theta = np.transpose(theta)
    inner = dot(theta,x)-y.T
    return (2*(dot(x, inner.T)))


def find_file_num(actor,add):
    file_num = 0
    for a in actor:
        dir = a+add
        for file in os.listdir(dir):
            file_num +=1
    return file_num


def set_creation(actor, Label,add):
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
    y = np.zeros((len(labels), len(labels[0])))
    for i in range(len(labels)):
        y[i] = labels[i]
    classfier = np.transpose(classfier)
    theta = np.zeros((1025, len(labels[0])))
    return classfier, y, theta


def mgrad_descent(f, df, x, y, init_t, alpha, max_iter):
    EPS = 1e-10   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        gradient = df(x, y, t,x.shape[0]).reshape(1025,6)
        t -= alpha*gradient
        if iter % 10000 == 0:
            print("Iter %i: cost = %.2f" % (iter, f(x, y, t)))
        iter += 1
    return t


def test(theta,answers,actors,add):
    cor = 0
    cnt = 0
    for actor in actors:
        i = 0
        dir = actor + add
        for file in os.listdir(dir):
            img = imread(dir+file).astype(float)
            img = img/255
            img = img.reshape((1,1024))
            img = np.hstack(([[1]], img))
            prediction = np.dot(img, theta)
            print("%s %i|pred:" % (actor, i))
            print(prediction)
            print("ans:")
            print(Label[actor])
            ans = Label[actor]
            pred = np.argmax(prediction)
            ans_index = np.argmax(ans)
            print 'ans'
            print ans_index
            print 'pre'
            print pred
            if pred == ans_index:
                cor+=1
                print 'correct'
            else:
                print 'incorrect'
            cnt += 1
        i+=1
    print 'score: '+ str(float(cor)/float(cnt))


# x, y, theta = set_creation(test_actor, Label)
# theta_p7 = mgrad_descent(f,df,x,y, theta, 0.000001, 150000)
# test(theta_p7,Label,test_actor)
#
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(theta_p7[1:, i].reshape((32,32)), cmap='RdBu')
#     plt.title(test_actor[i])
# plt.suptitle("Part 7 Theta")
# plt.show()





