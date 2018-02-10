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


def remove_duplicate():
    return




actor = ['Alec Baldwin', 'Steve Carell']

list = []


def find_max(i, max):
    if i > max:
        return i
    else:
        return max


def create_set(actor):
    max = 0
    for file in os.listdir('cropped/'):
        name = file.split('_')[1]
        if name == actor.lower():
            num = file.split('_')[-1]
            num = num.split('.')[0]
            num = int(num)
            max = find_max(num, max)
    t = np.arange(90)
    np.random.shuffle(t)
    return t[:90]

def create_folder(actor):
    for a in actor:
        if os.path.exists(a):
            shutil.rmtree(a)
        for retry in range(100):
            try:
                os.makedirs(a)
                break
            except:
                print "retry"
        os.makedirs(a+'/training')
        os.makedirs(a+'/validation')
        os.makedirs(a+'/testing')
        actor_last  = a.split(' ')[1].lower()
        list = create_set(actor_last)
        test = list[:10]
        val = list[10:20]
        train = list[-(len(list)-20):]
        for file in os.listdir('cropped/'):
            name = file.split('_')[1]
            if name == actor_last:
                num = file.split('_')[-1]
                num = num.split('.')[0]
                num = int(num)
                if num in test:
                    im = imread("cropped/" + file)
                    scipy.misc.imsave(a+'/testing/' + file, im)
                if num in val:
                    im = imread("cropped/" + file)
                    scipy.misc.imsave(a+'/validation/' + file, im)
                if num in train:
                    im = imread("cropped/" + file)
                    scipy.misc.imsave(a+'/training/' + file, im)

create_folder(actor)




