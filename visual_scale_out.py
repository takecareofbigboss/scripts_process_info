# Usage: python -m pdb visual_scale_out.py vgg-FDDB-v1-using-widerface-lr001_iter_40000.caffemodel \
#   /home/vis/dukang/data/WIDERFACE/WIDER_val/ WIDER_val/log_wider_val.txt 
#   score-scale-1,score-scale-2,score-scale-3,score-scale-4

import sys
import os
os.system('export PATH=/home/vis/common/anaconda/bin:$PATH')
os.system('source /home/vis/dukang/data/WIDERFACE/.bashrc_tangxu')
from PIL import Image
import numpy as np
import math 
#import scipy.io as sio
import matplotlib.pyplot as plt
caffe_root = '/home/vis/dukang/DenseBox_v7/'
sys.path.insert(0,caffe_root+'python')
import caffe
from caffe.proto import caffe_pb2
from PIL import Image 
import pprint,pickle
from struct import *
import random
import cv2
from math import *
import datetime


def swap(features_test_high,arr_tmp1,arr_tmp2,arr_tmp3):
	features_test_high_out = features_test_high
	features_test_high_out[:,:,0] = features_test_high[:,:,arr_tmp1]
	features_test_high_out[:,:,1] = features_test_high[:,:,arr_tmp2]
	features_test_high_out[:,:,2] = features_test_high[:,:,arr_tmp3]
	return features_test_high_out


def init_model():
  MODEL =  '/home/vis/dukang/faceDetect/models/fcn_detect_model/full_view/vgg_detect_v1_one_by_one_delete_scale4alone.prototxt'
  WEIGHT = '/home/vis/dukang/faceDetect/models/fcn_detect_model/full_view/'+caffemodel_name
  net = caffe.Net(MODEL,WEIGHT,caffe.TEST)
  return net

def parse_argv(layer_name_input):
  temp_list = []
  cut_str = layer_name_input.split(',')
  for i in range(len(cut_str)):
    temp_list.append(cut_str[i])
  return temp_list  


def get_array_of_image(img_path):
    # input_data = caffe.io.load_image(img_path)
    input_data = cv2.imread(img_path)
    input_data = np.array(input_data,dtype=np.float32)
    return input_data

def process_shape_devide_4(input_data):
    input_shape = input_data.shape
    w = input_shape[0]
    h = input_shape[1]
    if input_shape[0]%8!=0 or input_shape[1]%8!=0:
      w_size = int(floor((w+7.0)/8.0)*8)
      h_size = int(floor((h+7.0)/8.0)*8)
      w_bias = w_size - w
      h_bias = h_size - h
      array_img = np.zeros((w_size,h_size,3))
      array_img[0:w,0:h] = input_data
      input_data = array_img
    return input_data



def put_data_into_net(net,input_data):
  input_shape = []
  input_shape.append(1)
  input_shape.append(input_data.shape[2])
  input_shape.append(input_data.shape[0])
  input_shape.append(input_data.shape[1])
  transformer = caffe.io.Transformer({'data': input_shape})
  #transformer.set_raw_scale('data',255)
  transformer.set_transpose('data', (2,0,1))
  #transformer.set_channel_swap('data', (2,1,0))
  net.blobs['data'].reshape(input_shape[0],input_shape[1],input_shape[2],input_shape[3])
  net.blobs['data'].data[...] = transformer.preprocess('data', input_data)


def get_img_name(lines_input_txt):
  temp_txt = lines_input_txt.strip('\n\t\r')
  temp_txt = temp_txt.split('/')[-1]
  return temp_txt

def print_net_name(net):
  print 'The blobs (input,output) of your network -------------------------------------'
  for layer_name, blob in net.blobs.iteritems():
      print layer_name + '\t' + str(blob.data.shape)
  print 'The parameters of your network -----------------------------------------'
  for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

def process_array_list(features_scale_out):
  max_val = features_scale_out[0].max()
  min_val = features_scale_out[0].min()
  norm_feat = []
  for i in range(len(features_scale_out)):
    max_temp = features_scale_out[i].max()
    min_temp = features_scale_out[i].min()
    if max_temp >= max_val:
      max_val = max_temp
    if min_temp <= min_val:
      min_val = min_temp

  for i in range(len(features_scale_out)):
    float_array = (features_scale_out[i]-min_val)/(max_val-min_val)*255.0
    int_array = float_array.astype(int)
    norm_feat.append(int_array)

  return norm_feat


def read_txt_and_do_forward(net,test_img_file_dir,test_img_path,layer_name_input):
  with open(test_img_path, 'r') as f_dir:
    lines_input_txt = f_dir.readlines()

  for i in range(len(lines_input_txt)):
    starttime = datetime.datetime.now()
    temp_list = lines_input_txt[i].strip('\n\r\t')
    img_path = test_img_file_dir + temp_list
    print 'The system is processing the '+str(i)+' imgs-------------------------------'

    medium_1 = datetime.datetime.now()
    input_data = get_array_of_image(img_path)
    input_data = process_shape_devide_4(input_data)
    put_data_into_net(net,input_data)

    features_scale = []
    out = net.forward()
    
    medium_2 = datetime.datetime.now()
    saving_time = (medium_2 - medium_1).seconds
    print 'Loading Time is '+str(saving_time)

    features_scale_out = []
    for j in range(len(layer_name_input)):

      #print_net_name(net)
      features_scale = net.blobs[layer_name_input[j]].data[:].copy()

      shape_feat = features_scale.shape
      for k in range(shape_feat[0]):
        # features_scale_out.append(np.transpose(features_scale[k,:,:,:], [1,2,0]))
        features_scale_out.append(np.transpose(features_scale[k,:,:,:], [1,2,0])*255.0)

    #features_scale_out = process_array_list(features_scale_out)
    for j in range(len(layer_name_input)):
      img_gt_name = get_img_name(temp_list)
      cv2.imwrite('visual_scale_scale4alone_wo_norm/'+img_gt_name+'_4alone_wo_norm_'+str(j)+'.jpg', features_scale_out[j])

    endtime = datetime.datetime.now()
    print 'Running Time is '+str((endtime - starttime).seconds)

print 'processing the vec'

if __name__ == '__main__':
  if len(sys.argv) < 5:
    print ' Usage: python *.py caffemodel_name test_img_file_dir test_img_file layer_name_input'

  caffemodel_name = sys.argv[1]
  test_img_file_dir = sys.argv[2]
  test_img_path = sys.argv[3]
  layer_name_input = sys.argv[4]

  layer_name_input = parse_argv(layer_name_input)
  #layer_name_input = ['score-scale-1','score-scale-2','score-scale-3','score-scale-4']

  # load the caffemodel
  net = init_model()

  # read the input txt, and do the forward for xxx times.
  read_txt_and_do_forward(net,test_img_file_dir,test_img_path,layer_name_input)
  







