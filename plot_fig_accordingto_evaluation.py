import numpy as np
import scipy.io as sio
import time
from struct import *
import numpy as np
import cPickle as cp
import sys
import shutil
import os, sys, shutil
import random
import math
import h5py
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt

def read_two_lines(input_txt, flag_above_15k):
    f = open(input_txt, "r")
    orig_list = f.readlines()

    x = []
    y = []
    for i in range(len(orig_list)):
        if(flag_above_15k):
            str_tmp = orig_list[i].strip('\n\r\t')
            float_y = float(str_tmp.split(' ')[0])
            float_x = float(str_tmp.split(' ')[1])
            if float_x<15000:
                y = y + [float_y]
                x = x + [float_x]
        else:
            str_tmp = orig_list[i].strip('\n\r\t')
            y[i] = float(str_tmp.split(' ')[0])
            x[i] = float(str_tmp.split(' ')[1])            
    return x,y

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print ' Usage: python *.py input_txt.txt flag_above_15k'

    input_txt = sys.argv[1]
    flag_above_15k = int(sys.argv[2])

    x,y = read_two_lines(input_txt, flag_above_15k)

    # here, we plot x,y on the axies.
    plt.plot(x,y,'r-')
    plt.xlabel('false positive')
    plt.ylabel('recall')
    plt.margins(0.01)
    plt.subplots_adjust(bottom=0.15)
    plt.axis([0,15000,0,1])  
    plt.show()

    print 'Done.'
