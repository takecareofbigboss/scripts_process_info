import sys
import os
os.system('export PATH=/home/vis/common/anaconda/bin:$PATH')
from PIL import Image
import numpy as np
import math 
import scipy.io as sio
import matplotlib.pyplot as plt
caffe_root = '/home/vis/tangxu/caffe-gan/'
sys.path.insert(0,caffe_root+'python')
import caffe
from caffe.proto import caffe_pb2
from PIL import Image 
import pprint,pickle
from struct import *
import random
import cv2
         

def swap(features_test_high,arr_tmp1,arr_tmp2,arr_tmp3):
	features_test_high_out = features_test_high
	features_test_high_out[:,:,0] = features_test_high[:,:,arr_tmp1]
	features_test_high_out[:,:,1] = features_test_high[:,:,arr_tmp2]
	features_test_high_out[:,:,2] = features_test_high[:,:,arr_tmp3]
	return features_test_high_out


def init_model():
  MODEL =  '/home/vis/tangxu/caffe-gan/examples/age_synthesis/deploy_train_age_synthesis_cnn_dcnn_gan_v8.prototxt'
  WEIGHT = '/home/vis/tangxu/caffe-gan/examples/age_synthesis/'+caffemodel_name
  net = caffe.Net(MODEL,WEIGHT,caffe.TEST)
  return net


def get_array_of_image(img_path):
    input_data = caffe.io.load_image(img_path)
    input_data = caffe.io.resize(input_data,(108,108))
    input_data = np.array(input_data,dtype=np.float32)
    input_data = input_data * 255.0
    input_data = ((input_data-127.5)/127.5).astype(np.float32)
    return input_data


def put_data_into_net(net,input_data):
  transformer = caffe.io.Transformer({'data_pos': net.blobs['data_pos'].data.shape})
  transformer.set_transpose('data_pos', (2,0,1))
  transformer.set_channel_swap('data_pos', (2,1,0))
  net.blobs['data_pos'].reshape(1,3,108,108)
  net.blobs['data_pos'].data[...] = transformer.preprocess('data_pos', input_data)


def put_h5_into_net(net,tmp_age_split):
  blob_h5 = np.zeros([2,1,1])
  blob_h5 = blob_h5.astype(np.float32)
  blob_h5[0,0,0] = tmp_age_split[0]
  blob_h5[1,0,0] = tmp_age_split[1]
  net.blobs['data_h5'].data[...] = blob_h5


def give_str_to_array(use_age_group):
  use_age_group = use_age_group.split(',')
  all_use_age_group = []
  for i in range(len(use_age_group)):
    all_use_age_group.append(int(use_age_group[i]))
  return all_use_age_group


def read_txt_and_do_forward(net,test_img_path,use_age_group,layer_name_input):
  len_age = len(use_age_group) - 1
  for i in range(len_age):
    if i == 0:
      img_path = test_img_path
    else:
      img_path = 'synthesis_aging_images/img_aging_'+str(i)+'.jpg'
    tmp_age_split = [use_age_group[i],use_age_group[i+1]]
    input_data = get_array_of_image(img_path)
    put_data_into_net(net,input_data)
    put_h5_into_net(net,tmp_age_split)
   
    features_test_high = []
    out = net.forward()
    features_test_high=net.blobs[layer_name_input].data[:].copy()

    features_test_high = ((features_test_high+1)*127.5).astype(np.uint8)
    shape_feat = features_test_high.shape
    feat_10 = np.zeros((shape_feat[0],shape_feat[1],shape_feat[2],shape_feat[3]))
    for j in range(shape_feat[0]):
      features_test_high_tmp = np.transpose(features_test_high[j,:,:,:], [1,2,0])
      cv2.imwrite('synthesis_aging_images/img_aging_'+str(i+1)+'.jpg', features_test_high_tmp)



print 'processing the vec'

if __name__ == '__main__':
  if len(sys.argv) < 6:
    print ' Usage: python *.py iter_num layer_name caffemodel_name test_img_path use_age_group'

  iter_num = sys.argv[1]
  layer_name_input = sys.argv[2]
  caffemodel_name = sys.argv[3]
  test_img_path = sys.argv[4]
  use_age_group = sys.argv[5]
  use_age_group = give_str_to_array(use_age_group)

  # load the caffemodel
  net = init_model()

  # read the input txt, and do the forward for xxx times.
  read_txt_and_do_forward(net,test_img_path,use_age_group,layer_name_input)
  
