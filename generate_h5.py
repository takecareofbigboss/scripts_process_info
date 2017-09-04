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





def parse_txt_to_hdf5(shuffle_four_info,hdf5_m_n,split_part_num):
    print 'function parse_txt_to_hdf5() ...'

    with open(shuffle_four_info, 'r') as f_dir:
        lines_shuffle_four_info = f_dir.readlines()

    array_m_n = np.zeros((len(lines_shuffle_four_info),2,1,1))
    label_m_n = np.zeros((len(lines_shuffle_four_info),1,1,1))
    array_m_n = np.array(array_m_n,dtype=int)
    label_m_n = np.array(label_m_n,dtype=int)
    split_part_setting = math.ceil(len(lines_shuffle_four_info)/split_part_num)

    for i in range(len(lines_shuffle_four_info)):
        str_shuffle_info = lines_shuffle_four_info[i].strip('/r/n/t')
        int_shuffle_info = str_shuffle_info.split(' ')
        if i%50000==0:
            print i,'/',len(lines_shuffle_four_info)
        tmp_m_int = int(int_shuffle_info[2])
        tmp_n_int = int(int_shuffle_info[2])
        array_m_n[i,0,0,0] = tmp_m_int
        array_m_n[i,1,0,0] = tmp_n_int
        label_m_n[i,0,0,0] = tmp_n_int

    array_m_n = array_m_n.reshape((len(lines_shuffle_four_info),2,1,1))

    for i in range(split_part_num):
        start_index = i*split_part_setting
        end_index = (i+1)*split_part_setting-1
        str_hdf5_name = 'hdf5_part_same_'+str(i)+'.h5'
        print 'Saving hdf5 to ...',i,start_index,end_index
        with h5py.File(str_hdf5_name,'w') as f:
            f['data_h5_same'] = array_m_n[start_index:end_index]
            f['label_h5_same'] = label_m_n[start_index:end_index]
        print array_m_n[start_index]


if __name__ == '__main__':

    if len(sys.argv) < 9:
        print ' Usage: python *.py shuffle_four_info.txt hdf5_m_n.h5 split_part_num'

    shuffle_four_info = sys.argv[1]
    hdf5_m_n = sys.argv[2]
    split_part_num = int(sys.argv[3])

    parse_txt_to_hdf5(shuffle_four_info,hdf5_m_n,split_part_num)


    print 'Done.'
