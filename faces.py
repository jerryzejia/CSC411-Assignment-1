import input
import rescale_grey
import part2
import part3
import part5
import part6
import part7
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

# The code works on Windows 10, python 2.7, numpy 1.14

def Input():
    act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    input.get_uncroped(act)
    rescale_grey.crop(act)

def Part2():
    actor = ['Alec Baldwin', 'Steve Carell']
    part2.create_folder(actor)

def Part3():
    actor = ['Alec Baldwin', 'Steve Carell']
    Label = {'Alec Baldwin': -1, 'Steve Carell': 1}

    add = "/testing/"
    add_train = "/training/"
    x, y, theta = part3.set_creation(actor, Label, add_train)
    theta_p3 = part3.grad_descent(part3.f, part3.df, x, y, theta, 0.001, 20000)
    part3.test(theta_p3, Label, actor,add_train)
    print theta_p3.shape[0]

    x, y, theta = part3.set_creation(actor, Label, add_train)
    theta_p3 = part3.grad_descent(part3.f, part3.df, x, y, theta, 0.001, 20000)
    part3.test(theta_p3, Label, actor,add)

    #part 4
    plt.imshow(theta_p3[1:].reshape((32, 32)))
    plt.show()

    x, y, theta = part5.set_creation_limited(actor, Label,1)
    theta_p3 = part3.grad_descent(part3.f, part3.df, x, y, theta, 0.001, 20000)
    plt.imshow(theta_p3[1:].reshape((32, 32)))
    plt.show()

    x, y, theta = part3.set_creation(actor, Label,add)
    theta_p3 = part3.grad_descent(part3.f, part3.df, x, y, theta, 0.001, 3000)
    plt.imshow(theta_p3[1:].reshape((32, 32)))
    plt.show()

    theta_p3 = part3.grad_descent(part3.f, part3.df, x, y, theta, 0.001, 10)
    plt.imshow(theta_p3[1:].reshape((32, 32)))
    plt.show()

def Part5():
    test_actor = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    Label = {'Alec Baldwin': 1, 'Steve Carell': 1, 'Bill Hader': 1, 'Lorraine Bracco': -1, 'Peri Gilpin': -1,
             'Angie Harmon': -1}
    not_in_test_actor = ['Daniel Radcliffe', 'Michael Vartan','Gerard Butler','Kristin Chenoweth','Fran Drescher','America Ferrera']
    Label_not_in_actor = {'Daniel Radcliffe': 1, 'Michael Vartan' :1 ,'Gerard Butler':1,'Kristin Chenoweth':-1,'Fran Drescher':-1,'America Ferrera':-1}

    #uncomment the below section when testing
    # part2.create_folder(test_actor)
    # input.get_uncroped(not_in_test_actor)
    # rescale_grey.crop(not_in_test_actor)
    # part2.create_folder(not_in_test_actor)

    answer, answer_non, answer_notincluded = part5.data_create(test_actor, not_in_test_actor, Label, Label_not_in_actor)
    x_axis = [10,30,50,70]
    plt.plot(x_axis, answer, label = 'training')
    plt.plot(x_axis, answer_non, label = 'validation')
    plt.plot(x_axis, answer_notincluded, label = 'other 6 actor')

    plt.xlabel('#Number of Training image')
    plt.ylabel('%Correct')
    plt.title("Part 5 Overfitting ")
    plt.legend()
    plt.show()

def Part6():
    actor = ['Alec Baldwin']
    Label = {'Alec Baldwin': 1}
    x, y, theta = part5.set_creation_limited(actor, Label, 1)
    for i in theta:
        for j in i:
            j = random()
    new_theta = theta.copy()
    h = 0.0001
    dif = 0
    for k in range(5):
        i = np.random.choice(len(theta))
        j = np.random.choice(len(theta[0]))
        new_theta[i][j] = theta[i][j] + h
        dif += abs((part7.f(x, y, new_theta + theta) - part7.f(x, y, theta)) / h - part7.df(x, y, theta, 1025)[i][j])
    print(str(dif / 1025))

def Part7():
    test_actor = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    Label = {'Alec Baldwin': [1, 0, 0, 0, 0, 0], 'Steve Carell': [0, 1, 0, 0, 0, 0], 'Bill Hader': [0, 0, 1, 0, 0, 0],
             'Lorraine Bracco': [0, 0, 0, 1, 0, 0], 'Peri Gilpin': [0, 0, 0, 0, 1, 0],
             'Angie Harmon': [0, 0, 0, 0, 0, 1]}

    add_train = '/training/'
    x, y, theta = part7.set_creation(test_actor, Label,add_train)
    theta_p7 = part7.mgrad_descent(part7.f, part7.df, x, y, theta, 0.000001, 150000)
    part7.test(theta_p7, Label, test_actor,add_train)

    for i in range(theta.shape[1]):
        plt.imshow(theta_p7[1:, i].reshape((32, 32)), cmap='RdBu')
        plt.title(test_actor[i])
        plt.show()



    add = '/validation/'
    x, y, theta = part7.set_creation(test_actor, Label,add_train)
    theta_p7 = part7.mgrad_descent(part7.f, part7.df, x, y, theta, 0.000001, 150000)
    part7.test(theta_p7, Label, test_actor,add)


    #part 8



# Input()

#Part3()

# uncomment part of the code when testing part 5 if you need to download and crop the additional 6 actor
# Part5()

# Part6()

Part7()