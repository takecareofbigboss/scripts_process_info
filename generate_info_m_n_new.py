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
    return lines_input_txt,dict_input_txt


def pick_different_label_num(num_all_lines,lines_input_txt,key_name):
    rand_num = random.randint(0,num_all_lines-1)  
    tmp_line_txt = lines_input_txt[rand_num].strip('\n\t\r')
    start_name_info = tmp_line_txt.find('_',0)
    end_name_info = tmp_line_txt[:-7].rfind('_',)
    name_info = tmp_line_txt[start_name_info+1:end_name_info]
    while(name_info==key_name):
        rand_num = random.randint(0,num_all_lines-1)  
        tmp_line_txt = lines_input_txt[rand_num].strip('\n\t\r')
        start_name_info = tmp_line_txt.find('_',0)
        end_name_info = tmp_line_txt[:-7].rfind('_',)
        name_info = tmp_line_txt[start_name_info+1:end_name_info]
    return tmp_line_txt


def cross_m_n_all_pairs_and_find_neg(lines_input_txt,dict_input_txt,num_each_pair):
    all_pairs_with_path_label = []
    class_num = 0
    num_all_lines = len(lines_input_txt)

    for key_name in dict_input_txt.keys():
        print key_name
        print class_num
        size_key = len(dict_input_txt[key_name])
        for i in range(size_key):
            #print i
            pair_1 = dict_input_txt[key_name][i]
            age_1 = dict_input_txt[key_name][i][:2]
            for j in range(size_key):
                #print j
                pair_2 = dict_input_txt[key_name][j]
                age_2 = dict_input_txt[key_name][j][:2]
                if int(age_2) > int(age_1):  
                    for k in range(num_each_pair):
                        same_pick_img = pair_2
                        rand_pick_img = pick_different_label_num(num_all_lines,lines_input_txt,key_name)
                        pair_list = [pair_1, rand_pick_img, age_1, age_2, str(class_num), pair_2]
                        all_pairs_with_path_label.append(pair_list)
        class_num += 1
    return all_pairs_with_path_label


def print_lines_to_txt(lines_four_info,four_info):
    print 'function print_lines_to_txt() ...'
    file_all = open(four_info, 'w')
    for i in range(len(lines_four_info)):
        if i%50000==0:
            print i,'/',len(lines_four_info)
        line_str = lines_four_info[i][0]+' '+lines_four_info[i][1]+' '+lines_four_info[i][2]+' '+lines_four_info[i][3]+' '+str(lines_four_info[i][4])+' '+lines_four_info[i][5]
        file_all.writelines(line_str+'\n')
    file_all.close()


def shuffle_lines_of_txt(lines_four_info,shuffle_four_info):
    print 'function shuffle_lines_of_txt() ...'
    file_all = open(shuffle_four_info, 'w')
    sorted_array = range(len(lines_four_info))
    shuffle_array = sorted_array
    random.shuffle(shuffle_array)
    shuffle_lines_four_info = []

    for i in range(len(lines_four_info)):
        shuffle_lines_four_info.append(lines_four_info[shuffle_array[i]])

    for i in range(len(shuffle_lines_four_info)):
        if i%50000==0:
            print i,'/',len(shuffle_lines_four_info)
        line_str = shuffle_lines_four_info[i][0]+' '+shuffle_lines_four_info[i][1]+' '+shuffle_lines_four_info[i][2]+' '+shuffle_lines_four_info[i][3]+' '+str(shuffle_lines_four_info[i][4]+' '+shuffle_lines_four_info[i][5])
        file_all.writelines(line_str+'\n')
    file_all.close()

    return shuffle_lines_four_info


def read_txt_to_image_label_txt(shuffle_lines_four_info,shuffle_four_info,first_file,second_file,third_file):
    print 'function read_txt_to_image_label_txt() ...'

    file_all = open(first_file, 'w')
    for i in range(len(shuffle_lines_four_info)):
        if i%50000==0:
            print i,'/',len(shuffle_lines_four_info)
        line_str = shuffle_lines_four_info[i][0]+' '+str(shuffle_lines_four_info[i][4])
        file_all.writelines(line_str+'\n')
    file_all.close()

    file_all = open(second_file, 'w')
    for i in range(len(shuffle_lines_four_info)):
        if i%50000==0:
            print i,'/',len(shuffle_lines_four_info)
        line_str = shuffle_lines_four_info[i][1]+' '+str(shuffle_lines_four_info[i][4])
        file_all.writelines(line_str+'\n')
    file_all.close()

    file_all = open(third_file, 'w')
    for i in range(len(shuffle_lines_four_info)):
        if i%50000==0:
            print i,'/',len(shuffle_lines_four_info)
        line_str = shuffle_lines_four_info[i][5]+' '+str(shuffle_lines_four_info[i][4])
        file_all.writelines(line_str+'\n')
    file_all.close()


def parse_txt_to_hdf5(shuffle_lines_four_info,shuffle_four_info,hdf5_m_n,split_part_num):
    print 'function parse_txt_to_hdf5() ...'

    array_m_n = np.zeros((len(shuffle_lines_four_info),2,1,1))
    label_m_n = np.zeros((len(shuffle_lines_four_info),1,1,1))
    array_m_n = np.array(array_m_n,dtype=int)
    label_m_n = np.array(label_m_n,dtype=int)
    split_part_setting = math.ceil(len(shuffle_lines_four_info)/split_part_num)

    for i in range(len(shuffle_lines_four_info)):
        if i%50000==0:
            print i,'/',len(shuffle_lines_four_info)
        tmp_m_int = int(shuffle_lines_four_info[i][2])
        tmp_n_int = int(shuffle_lines_four_info[i][3])
        array_m_n[i,0,0,0] = tmp_m_int
        array_m_n[i,1,0,0] = tmp_n_int
        label_m_n[i,0,0,0] = tmp_n_int

    array_m_n = array_m_n.reshape((len(shuffle_lines_four_info),2,1,1))

    for i in range(split_part_num):
        start_index = i*split_part_setting
        end_index = (i+1)*split_part_setting-1
        str_hdf5_name = 'hdf5_part_'+str(i)+'.h5'
        print 'Saving hdf5 to ...',i,start_index,end_index
        with h5py.File(str_hdf5_name,'w') as f:
            f['data_h5'] = array_m_n[start_index:end_index]
            f['label_h5'] = label_m_n[start_index:end_index]
        print array_m_n[start_index]


if __name__ == '__main__':

    if len(sys.argv) < 9:
        print ' Usage: python *.py input_txt.txt four_info.txt num_each_pair shuffle_four_info.txt \
            first_file.txt second_file.txt third_file.txt hdf5_m_n.h5 split_part_num'

    input_txt = sys.argv[1]
    four_info = sys.argv[2]
    num_each_pair = int(sys.argv[3])
    shuffle_four_info = sys.argv[4]
    first_file = sys.argv[5]
    second_file = sys.argv[6]
    third_file = sys.argv[7]
    hdf5_m_n = sys.argv[8]
    split_part_num = int(sys.argv[9])

    lines_input_txt,dict_input_txt = put_txt_in_dict(input_txt)
    lines_four_info = cross_m_n_all_pairs_and_find_neg(lines_input_txt,dict_input_txt,num_each_pair)
    print_lines_to_txt(lines_four_info,four_info)
    shuffle_lines_four_info = shuffle_lines_of_txt(lines_four_info,shuffle_four_info)
    read_txt_to_image_label_txt(shuffle_lines_four_info,shuffle_four_info,first_file,second_file,third_file)
    parse_txt_to_hdf5(shuffle_lines_four_info,shuffle_four_info,hdf5_m_n,split_part_num)


    print 'Done.'
