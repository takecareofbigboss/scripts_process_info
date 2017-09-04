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

    for i in range(len(lines_input_txt)):
        if i == 1049:
            aa = 1
        tmp_line_txt = lines_input_txt[i].strip('\n\t\r')
        age_info = tmp_line_txt.split('_')[0]
        start_name_info = tmp_line_txt.find('_',0)
        end_name_info = tmp_line_txt.rfind('_')
        name_info = tmp_line_txt[start_name_info+1:end_name_info]

        if name_info not in dict_input_txt:
            dict_input_txt[name_info] = []
        if name_info in dict_input_txt:
            dict_input_txt[name_info].append(tmp_line_txt)
    return dict_input_txt


def put_dict_to_array(dict_input_txt):
    all_pairs_with_path_label = []
    class_num = 0
    for key_name in dict_input_txt.keys():
        print key_name
        size_key = len(dict_input_txt[key_name])
        for i in range(size_key):
            pair_1 = dict_input_txt[key_name][i]
            age_1 = dict_input_txt[key_name][i][:2]
            for j in range(size_key):
                age_2 = str(int(age_1) + 1)
                if age_2+'_' in dict_input_txt[key_name][j]:
                    pair_2 = dict_input_txt[key_name][j]
                    pair_list = [pair_1, pair_2, class_num]
                    all_pairs_with_path_label.append(pair_list)
        class_num += 1
    return all_pairs_with_path_label


def put_array_pairs_to_txt(all_pairs_with_path_label,age_start,age_end,out_1,out_2,out_all):
    str_1 = str(age_start)+'_to_'+str(age_end)+'_'+out_1
    str_2 = str(age_start)+'_to_'+str(age_end)+'_'+out_2
    str_all = str(age_start)+'_to_'+str(age_end)+'_'+out_all

    file_1 = open(str_1, 'w')
    file_2 = open(str_2, 'w')
    file_all = open(str_all, 'w')

    for line_array in all_pairs_with_path_label:
        age_1 = int(line_array[0][:2])
        age_2 = int(line_array[1][:2])
        if age_1 in range(age_start,age_end+1):
            file_1.writelines(line_array[0]+' '+str(line_array[2])+'\n')
            file_2.writelines(line_array[1]+' '+str(line_array[2])+'\n')
            file_all.writelines(line_array[0]+' '+line_array[1]+' '+str(line_array[2])+'\n')

    file_1.close()
    file_2.close()
    file_all.close()




if __name__ =='__main__':
    
    if len(sys.argv) < 6:
        print ' Usage: python *.py input_txt.txt age_group out_1.txt out_2.txt out_all.txt'

    input_txt = sys.argv[1]
    age_group = sys.argv[2] 
    out_1 = sys.argv[3]
    out_2 = sys.argv[4]
    out_all = sys.argv[5]

    age_start = int(age_group.split(',')[0])
    age_end = int(age_group.split(',')[1])

    dict_input_txt = put_txt_in_dict(input_txt)
    all_pairs_with_path_label = put_dict_to_array(dict_input_txt)
    put_array_pairs_to_txt(all_pairs_with_path_label,age_start,age_end,out_1,out_2,out_all)


    print 'Done.'
