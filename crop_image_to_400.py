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

def crop_input_to_output(input_txt,input_dir,output_dir):
    iter_num = 0
    for line_test in open(input_txt):
        str_img = line_test.strip('\n\t\r')
        im = Image.open(input_dir+'/'+str_img)
        draw = ImageDraw.Draw(im)
        im_array = np.array(im)
        im_array = im_array[100:500,100:500,:]
        im = Image.fromarray(im_array)
        
        save_path = output_dir+'/'+str_img
        dir_path = output_dir+'/'+str_img.split('/')[0]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)               
        im.save(save_path)
        iter_num += 1
        print iter_num



if __name__ == '__main__':

    if len(sys.argv) < 4:
        print ' Usage: python *.py input_txt.txt input_dir output_dir'

    input_txt = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]

    crop_input_to_output(input_txt,input_dir,output_dir)

    print 'Done.'
