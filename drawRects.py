#!/usr/bin/python
import os
import sys
import os.path 
import shutil
import random
import cv2
import math
import numpy as np

if __name__ =='__main__':

    gtFile      = sys.argv[1]
    srcDir      = sys.argv[2]
    dstDir      = sys.argv[3]
    if not os.path.exists(dstDir):
        os.system("mkdir -p "+dstDir)
        
    with open(gtFile,'r') as f:
        lines = f.readlines()

    count = 0 
    line_index = 0
    while line_index < len(lines):
        count += 1
        if 0 == count % 50:
            print '%s/%s'%(count,len(lines))

        path = lines[line_index].strip('\n\t')+'.jpg'
        line_index += 1


        numFace = int( lines[line_index].strip('\n\t') )
        line_index += 1

        srcPath = srcDir + path
        img = cv2.imread(srcPath)
        if img is None:
            print '%s is not be found.'%(path)
            line_index += numFace
        for face_index in range(0,numFace):
            words = lines[line_index].strip('\n\t').split()
            line_index += 1
            if not len(words) == 5 :
                continue
            lt_x = int( float(words[0])+0.5 )
            lt_y = int( float(words[1])+0.5 )
            rb_x = lt_x + int( float(words[2])+0.5 )
            rb_y = lt_y + int( float(words[3])+0.5 )
            cv2.rectangle(img,(lt_x,lt_y),(rb_x,rb_y),(255,0,0),1)
            
        savePath = dstDir + path[path.find('/'):]
        saveDir = savePath[0:savePath.rfind('/')]
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        cv2.imwrite(savePath,img)

