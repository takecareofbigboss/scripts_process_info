import numpy as np
import scipy.io as sio
import time
from struct import *
import numpy as np
import cPickle as cp
import sys
import shutil
import os,sys,shutil


def put_txt_in_dict(input_txt):
    with open(input_txt, 'r') as f_dir:
        lines_input_txt = f_dir.readlines()

    dict_input_txt = {}
    num_class = 0
    for i in range(len(lines_input_txt)):
        tmp_line_txt = lines_input_txt[i].strip('\n\t\r')
        if 'WIDER_val/images/' in tmp_line_txt:
            if i != 0:
                num_class += 1
            dict_input_txt[num_class] = []
            dict_name = tmp_line_txt
            dict_input_txt[num_class].append(tmp_line_txt)
        if 'WIDER_val/images/' not in tmp_line_txt:
            if len(tmp_line_txt) > 6:
                tmp_line_txt = tmp_line_txt[:-2]
                split_str = tmp_line_txt.split(' ')
                x1_val = float(split_str[0])
                y1_val = float(split_str[1])
                x3_val = float(split_str[4])
                y3_val = float(split_str[5])    
                cx_val = float((x3_val - x1_val)/2 + x1_val)        
                cy_val = float((y3_val - y1_val)/2 + y1_val)  
                tmp_line_txt = tmp_line_txt + str(cx_val) + ' ' + str(cy_val) + ' '     
                dict_input_txt[num_class].append(tmp_line_txt)

    return dict_input_txt


def make_the_same_str(int_num):
    str_out = ''
    for i in range(int_num):
        str_out = str_out + '0.0 '

    return str_out


def append_format2txt(str_out,dict_input_txt,output_txt):
    file_all = open(output_txt, 'w')
    for i in range(len(dict_input_txt)):   
        line_str = dict_input_txt[i][0] + ' '
        for j in range(1,len(dict_input_txt[i])):
            line_str = line_str + str_out
            line_str = line_str + dict_input_txt[i][j]
        file_all.writelines(line_str+'\n')
    file_all.close()


if __name__ =='__main__':
    
    if len(sys.argv) < 3:
        print ' Usage: python *.py input_txt.txt output_txt.txt'

    input_txt = sys.argv[1]
    output_txt = sys.argv[2] 

    dict_input_txt = put_txt_in_dict(input_txt)
    str_out = make_the_same_str(144)
    append_format2txt(str_out,dict_input_txt,output_txt)

    


    print 'Done.'
