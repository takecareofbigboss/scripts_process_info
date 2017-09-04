import numpy as np
import time
from struct import *
import numpy as np
import cPickle as cp
import sys
import shutil
import os, sys, shutil
import random
import math
from PIL import Image,ImageDraw

def crop_input_to_output(input_txt,input_dir,output_dir):
    iter_num = 0
    for line_test in open(input_txt):
        str_img = line_test.strip('\n\t\r')   
        input_name = input_dir+'/'+str_img
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)   
        output_name = output_dir+'/'+str_img+'_pred.jpg'         
        shutil.copy(input_name,output_name)
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
