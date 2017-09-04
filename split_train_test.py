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


def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list


def put_txt_in_dict(input_txt):
    with open(input_txt, 'r') as f_dir:
        lines_input_txt = f_dir.readlines()

    dict_input_txt = {}

    for i in range(len(lines_input_txt)):
        if i == 1049:
            aa = 1
        tmp_line_txt = lines_input_txt[i].strip('\n\t\r')
        age_info = tmp_line_txt.split('_')[0]
        start_name_info = tmp_line_txt.find('_',0)
        end_name_info = tmp_line_txt[:-7].rfind('_',)
        name_info = tmp_line_txt[start_name_info+1:end_name_info]

        if name_info not in dict_input_txt:
            dict_input_txt[name_info] = []
        if name_info in dict_input_txt:
            dict_input_txt[name_info].append(tmp_line_txt)
    return dict_input_txt


def split_to_folders(dict_input_txt,num_test,train_dir,test_dir):
	rand_200 = random_int_list(0,len(dict_input_txt)-1,num_test)
	class_num = 0
	for key_name in dict_input_txt.keys():
		if class_num in rand_200:
			print 'Testing: ',key_name
			size_key = len(dict_input_txt[key_name])
			for i in range(size_key):
				img_name = dict_input_txt[key_name][i]
				path_name = 'CACD_cropped/'+img_name
				shutil.copy(path_name,test_dir)
		else:
			print 'Training: ',key_name
			size_key = len(dict_input_txt[key_name])
			for i in range(size_key):
				img_name = dict_input_txt[key_name][i]
				path_name = 'CACD_cropped/'+img_name
				shutil.copy(path_name,train_dir)
		class_num += 1


if __name__ == '__main__':

    if len(sys.argv) < 5:
        print ' Usage: python *.py input_txt.txt num_test train_dir test_dir'

    input_txt = sys.argv[1]
    num_test = int(sys.argv[2])
    train_dir = sys.argv[3]
    test_dir = sys.argv[4]

    tmp_cmd = 'mkdir -p '+train_dir
    os.system(tmp_cmd)
    tmp_cmd = 'mkdir -p '+test_dir
    os.system(tmp_cmd)

    dict_input_txt = put_txt_in_dict(input_txt)
    split_to_folders(dict_input_txt,num_test,train_dir,test_dir)


    print 'Done.'
