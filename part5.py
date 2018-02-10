import part2
import part3
import input
import rescale_grey
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
Label = {'Alec Baldwin': 1, 'Steve Carell':1, 'Bill Hader':1, 'Lorraine Bracco':-1, 'Peri Gilpin':-1, 'Angie Harmon':-1}


# part2.create_folder(test_actor)
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
            labels.append(int(Label[a]))
            i+=1
            j+=1
            if j == file_num:
                break
    y = np.zeros((len(labels), 1))
    for i in range(len(labels)):
        y[i] = labels[i]
    classfier = np.transpose(classfier)
    theta = np.zeros((1025, 1))

    return classfier, y, theta
def test(theta,Label,actor, keyword):
    cor = 0
    cnt = 0
    for a in actor:
        i = 0
        dir = a + keyword
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
    return (float(cor)/float(cnt))



# x,y,theta = part3.set_creation(test_actor, Label)
# x,y,theta = set_creation_limited(test_actor, Label, 30)
# theta_p5 = part3.grad_descent(part3.f, part3.df, x, y, theta, 0.0001, 10000)
# part3.test(theta_p5,Label,test_actor)
# print('===========================================================================================================================')


not_in_test_actor = ['Daniel Radcliffe', 'Michael Vartan','Gerard Butler','Kristin Chenoweth','Fran Drescher','America Ferrera']
Label_not_in_actor = {'Daniel Radcliffe': 1, 'Michael Vartan' :1 ,'Gerard Butler':1,'Kristin Chenoweth':-1,'Fran Drescher':-1,'America Ferrera':-1}

# input.get_uncroped(not_in_test_actor)
# rescale_grey.crop(not_in_test_actor)
# part2.create_folder(not_in_test_actor)
# answer =[]
# answer_non = []
# x,y,theta = set_creation_limited(test_actor, Label, 10)
# theta_p5 = part3.grad_descent(part3.f, part3.df, x, y, theta, 0.0001, 10000)
# ans_10 = test(theta_p5,Label_not_in_actor,not_in_test_actor, "/training/")
# ans10 = test(theta_p5,Label_not_in_actor,not_in_test_actor, "/testing/")
# answer_non.append(ans10*100)
# answer.append(ans_10*100)
#
# x,y,theta = set_creation_limited(test_actor, Label, 30)
# theta_p5 = part3.grad_descent(part3.f, part3.df, x, y, theta, 0.0001, 10000)
# ans_30 =test(theta_p5,Label_not_in_actor,not_in_test_actor, "/training/")
# ans30 = test(theta_p5,Label_not_in_actor,not_in_test_actor, "/testing/")
# answer_non.append(ans30*100)
# answer.append(ans_30*100)
#
# x,y,theta = set_creation_limited(test_actor, Label, 50)
# theta_p5 = part3.grad_descent(part3.f, part3.df, x, y, theta, 0.0001, 10000)
# ans_50 =test(theta_p5,Label_not_in_actor,not_in_test_actor, "/training/")
# ans50 = test(theta_p5,Label_not_in_actor,not_in_test_actor, "/testing/")
# answer.append(ans_50*100)
# answer_non.append(ans50*100)
#
# x,y,theta = part3.set_creation(test_actor, Label)
# theta_p5 = part3.grad_descent(part3.f, part3.df, x, y, theta, 0.0001, 10000)
# ans_60 =test(theta_p5,Label_not_in_actor,not_in_test_actor, "/training/")
# ans60 = test(theta_p5,Label_not_in_actor,not_in_test_actor, "/testing/")
# answer.append(ans_60*100)
# answer_non.append(ans60*100)
#
#
#
# print('===========================================================================================================================')
#
# x_axis = [10,30,50,70]
# plt.plot(x_axis, answer, label = 'validation')
# plt.plot(x_axis, answer_non, label = 'training')
# plt.xlabel('#Number of Training image')
# plt.ylabel('%Correct')
# plt.title("Part 5 Overfitting ")
# plt.legend()
# plt.show()
# print('===========================================================================================================================')
#
def data_create(test_actor,not_in_test_actor,Label,Label_not_in_actor):
    part2.create_folder(not_in_test_actor)
    answer =[]
    answer_non = []
    answer_notincluded = []
    x,y,theta = set_creation_limited(test_actor, Label, 10)
    theta_p5 = part3.grad_descent(part3.f, part3.df, x, y, theta, 0.001, 10000)
    ans_10 = test(theta_p5,Label,test_actor, "/training/")
    ans10 = test(theta_p5,Label,test_actor, "/testing/")
    ans_not = test(theta_p5,Label,test_actor, "/testing/")
    ans_not =test(theta_p5,Label_not_in_actor,not_in_test_actor, "/testing/")
    answer_non.append(ans10*100)
    answer.append(ans_10*100)
    answer_notincluded.append(ans_not*100)
    x,y,theta = set_creation_limited(test_actor, Label, 30)
    theta_p5 = part3.grad_descent(part3.f, part3.df, x, y, theta, 0.001, 10000)
    ans_30 =test(theta_p5,Label,test_actor, "/training/")
    ans30 = test(theta_p5,Label,test_actor, "/testing/")
    ans_not =test(theta_p5,Label_not_in_actor,not_in_test_actor, "/testing/")

    answer_non.append(ans30*100)
    answer.append(ans_30*100)
    answer_notincluded.append(ans_not*100)

    x,y,theta = set_creation_limited(test_actor, Label, 50)
    theta_p5 = part3.grad_descent(part3.f, part3.df, x, y, theta, 0.001, 10000)
    ans_50 =test(theta_p5,Label,test_actor, "/training/")
    ans50 = test(theta_p5,Label,test_actor, "/testing/")
    ans_not =test(theta_p5,Label_not_in_actor,not_in_test_actor, "/testing/")
    answer.append(ans_50*100)
    answer_non.append(ans50*100)
    answer_notincluded.append(ans_not*100)
    add = "/training/"
    x,y,theta = part3.set_creation(test_actor, Label,add)
    theta_p5 = part3.grad_descent(part3.f, part3.df, x, y, theta, 0.001, 10000)
    ans_60 =test(theta_p5,Label,test_actor, "/training/")
    ans60 = test(theta_p5,Label,test_actor, "/testing/")
    ans_not =test(theta_p5,Label_not_in_actor,not_in_test_actor, "/testing/")
    answer.append(ans_60*100)
    answer_non.append(ans60*100)
    answer_notincluded.append(ans_not*100)
    print answer_notincluded
    return answer, answer_non, answer_notincluded
#
#
# x_axis = [10,30,50,70]
# plt.plot(x_axis, answer, label = 'training')
# plt.plot(x_axis, answer_non, label = 'validation')
# plt.plot(x_axis, answer_notincluded, label = 'other 6 actor')
#
# plt.xlabel('#Number of Training image')
# plt.ylabel('%Correct')
# plt.title("Part 5 Overfitting ")
# plt.legend()
# plt.show()
