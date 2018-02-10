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



act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']


def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray / 255.

def crop(actor):
    for a in actor:
        name = a.split()[1].lower()
        i = 0
        for file in os.listdir('uncropped/'):
            for line in open("faces_subset.txt"):
                file_num = file.split('_')[1]
                file_num = file_num.split('.')[0]
                if a in line and file_num == line.split()[3]:

                    size = line.split()[5]
                    x1 = int(size.split(',')[0])
                    y1 = int(size.split(',')[1])
                    x2 = int(size.split(',')[2])
                    y2 = int(size.split(',')[3])

                    extension = line.split()[4].split('.')[-1]
                    filename = name +'_'+ line.split()[3] + '.' + extension
                    if extension == '.jpg':
                        extension = '.jpeg'
                    try:
                        im = imread("uncropped/" + filename)
                    except IOError as e:
                        extension = 'jpg'
                    try:
                        im = imread("uncropped/" + filename)
                    except:
                        continue

                    im_crop = im[y1:y2, x1:x2]
                    try:
                        im_resize = imresize(im_crop,(32,32))
                    except:
                        print filename
                        im_resize = im_crop

                    try: im_resize = rgb2gray(im_resize)
                    except:
                        print filename
                    filename = name +'_'+ line.split()[3]+'_'+ str(i) + '.' + extension
                    i=i+1
                    try:
                        scipy.misc.imsave('cropped/cropped_' + filename, im_resize)
                    except:
                        print filename
    return