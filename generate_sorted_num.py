# -*- coding: utf-8 -*-
"""
Created on 19-07-2017  
@author: Xu Tang
Usage: used for generating the sorted array.
"""

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

def get_sorted_array(begin_num, end_num):
	"""
	Input: 
		begin_num: the beginning of the array.
		end_num: the ending of the array.
	Output:
		The sorted array.
	"""

	sorted_array = range(begin_num, end_num)
	return sorted_array



def read_array_into_txt(sorted_array, output_txt):
	"""
	Input: 
		sorted_array: The sorted array need to be write into our txt.
	Output:
		output_txt: Write all of them into our txt.
	"""

	output_txt_handler = open(output_txt,"w")
	for i in range(len(sorted_array)):
		output_txt_handler.writelines(str(sorted_array[i])+'\n')
	output_txt_handler.close()
	
if __name__ == '__main__':

    if len(sys.argv) < 4:
        print ' Usage: python *.py begin_num end_num output_txt'

    begin_num = int(sys.argv[1])
    end_num = int(sys.argv[2])
    output_txt = sys.argv[3]

    sorted_array = get_sorted_array(begin_num, end_num)
    read_array_into_txt(sorted_array, output_txt)

    print 'Done.'

