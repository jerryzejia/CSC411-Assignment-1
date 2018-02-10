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
list = []

def timeout(func, args=(), kwargs={}, timeout_duration=30, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()



# Note: you need to create the uncropped folder first in order
# for this to work
def get_uncroped(act):
    for a in act:
        name = a.split()[1].lower()
        for line in open("faces_subset.txt"):
            if a in line:
                if not os.path.exists("uncropped"):
                    os.makedirs('uncropped')
                filename = name +'_' +line.split()[3]+ '.' + line.split()[4].split('.')[-1]
                # A version without timeout (uncomment in case you need to
                # unsupress exceptions, which timeout() does)
                # testfile.retrieve(line.split()[4], "uncropped/"+filename)
                # timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/" + filename), {}, 30)
                if not os.path.isfile("uncropped/"+filename):
                    continue
                print filename

    return