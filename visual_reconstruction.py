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

print 'processing the vec'

if __name__ == '__main__':
  if len(sys.argv) < 3:
    print ' Usage: python *.py iter_num layer_name caffemodel_name'

  iter_num = sys.argv[1]
  layer_name_input = sys.argv[2]
  caffemodel_name = sys.argv[3]

  # load the caffemodel
  MODEL =  '/home/vis/tangxu/caffe-gan/examples/age_synthesis/test_age_synthesis_cnn_dcnn.prototxt'
  WEIGHT = '/home/vis/tangxu/caffe-gan/examples/age_synthesis/'+caffemodel_name
  net = caffe.Net(MODEL,WEIGHT,caffe.TEST)
  features_test_high = []
  out = net.forward()

  # for each layer, show the output shape
  for layer_name, blob in net.blobs.iteritems():
  	print layer_name + '\t' + str(blob.data.shape)
  for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

  features_test_high=net.blobs[layer_name_input].data[:].copy()

  features_test_high = ((features_test_high+1)*127.5).astype(np.uint8)
  shape_feat = features_test_high.shape
  feat_10 = np.zeros((shape_feat[0],shape_feat[1],shape_feat[2],shape_feat[3]))
  for i in range(shape_feat[0]):
	features_test_high_tmp = np.transpose(features_test_high[i,:,:,:], [1,2,0])
	cv2.imwrite('synthesis_images/image_'+str(i)+'_'+str(iter_num)+'.jpg', features_test_high_tmp)
