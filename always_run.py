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
from PIL import Image,ImageDraw

def while_keep(wait_seconds):
    while 1>0:
        time.sleep(wait_seconds)
        str_cmd = 'echo \'hello\''
        os.system(str_cmd)


if __name__ == '__main__':

    if len(sys.argv) < 4:
        print ' Usage: python *.py wait_seconds'

    wait_seconds = int(sys.argv[1])

    while_keep(wait_seconds)

    print 'Done.'
