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


hdf5_file_name = sys.argv[1]
file    = h5py.File(hdf5_file_name, 'r') 
dataset = file['data_h5']
arr1ev  = dataset[0]
file.close()