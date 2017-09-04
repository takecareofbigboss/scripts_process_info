#  python -m pdb read_txt_cut_threshold.py log_scale4alone_orig.txt \
#      ./pred_VGG_finetune_fddb_wo_parsing_one_by_one_delete_scale4alone \
#      ./pred_VGG_finetune_fddb_wo_parsing_one_by_one_delete_scale4alone_threshold045 0.45

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


def crop_input_to_output(input_txt,input_dir,output_dir,thres_score):
    iter_num = 0
    for line_test in open(input_txt):
        str_txt = line_test.strip('\n\t\r')   
        input_name = input_dir+'/'+str_txt
        output_dir_final = output_dir+'/'+str_txt[:str_txt.find('/')]
        if not os.path.exists(output_dir_final):
            os.makedirs(output_dir_final)   
        output_name = output_dir+'/'+str_txt   
        output_txt_handler = open(output_name,"w")
        ignore_num = 0    
        with open(input_name, 'r') as f_dir:
            lines_input_txt = f_dir.readlines()
        for j in range(len(lines_input_txt)):
            list_lines = lines_input_txt[j].strip('\n\r\t').split(' ')
            if len(list_lines)>=5:
                score = float(lines_input_txt[j].strip('\n\r\t').split(' ')[-1])
                if score<=thres_score:
                    ignore_num += 1
                    continue
            output_txt_handler.writelines(str(lines_input_txt[j].strip('\n\r\t')+'\n'))
        output_txt_handler.close()

        num_face = int(lines_input_txt[1].strip('\n\r\t'))
        left_num_face = str(num_face-ignore_num)

        f_in = open(output_name,'r')
        line_f_in = f_in.readlines()
        line_f_in[1] = str(left_num_face)+'\n'

        f_out = open(output_name,'w')
        f_out.writelines(line_f_in)
        f_out.close() 

        iter_num += 1
        print iter_num


if __name__ == '__main__':

    if len(sys.argv) < 4:
        print ' Usage: python *.py input_txt.txt input_dir output_dir threshold'

    input_txt = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]
    thres_score = float(sys.argv[4])

    crop_input_to_output(input_txt,input_dir,output_dir,thres_score)

    print 'Done.'






