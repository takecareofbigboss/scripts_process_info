#!/usr/bin/python
# -*- coding: cp936 -*-
import os
import shutil
import cv2
import urllib2
import numpy as np
from urllib2 import urlopen
from cStringIO import StringIO

def getRect( face):
    #print face[0],face[1],face[2],face[3],face[4],face[5],face[6],face[7]
    xmin = min(face[0],face[2],face[4],face[6])
    ymin = min(face[1],face[3],face[5],face[7])
    xmax = max(face[0],face[2],face[4],face[6])
    ymax = max(face[1],face[3],face[5],face[7])

    return [xmin,ymin,xmax,ymax]

def change_gt_format(face_gt_array):
    x1 = face_gt_array[0]    
    y1 = face_gt_array[1]
    x2 = face_gt_array[0] + face_gt_array[2]
    y2 = face_gt_array[1]
    x3 = face_gt_array[0] + face_gt_array[2]
    y3 = face_gt_array[1] + face_gt_array[3]
    x4 = face_gt_array[0]
    y4 = face_gt_array[1] + face_gt_array[3]
    return [x1,y1,x2,y2,x3,y3,x4,y4]

def getOverlap(rect1, rect2):
    lt_x = max(rect1[0], rect2[0])
    lt_y = max(rect1[1], rect2[1])
    rb_x = min(rect1[2], rect2[2])
    rb_y = min(rect1[3], rect2[3])
    
    area1 = (rect1[2]-rect1[0])*(rect1[3]-rect1[1])
    area2 = (rect2[2]-rect2[0])*(rect2[3]-rect2[1])

    if (rb_x > lt_x) and (rb_y > lt_y):
        intersection = (rb_x - lt_x)*(rb_y - lt_y)
    else:
        return 0

    
    intersection = min( intersection, area1, area2)
    union = area1 + area2 - intersection
    return( float(intersection)/union)
    
    #return( float(intersection)/area1)

def isSameFace(face_gt,face_pred):
    rect_gt = getRect(face_gt)
    rect_pred = getRect(face_pred)

    overlap = getOverlap(rect_gt, rect_pred)

    return (overlap >= 0.3)
    
    
def drawRotateRectange(img,face,color,thickness):
    for i in range(4):
        cv2.line(img,(int(face[(2*i)%8]),int(face[(2*i+1)%8])), (int(face[(2*i+2)%8]),int(face[(2*i+3)%8])),color,thickness)

def isSmallFace(face):
    widthsquare = (face[0]-face[2])*(face[0]-face[2]) + (face[1]-face[3])*(face[1]-face[3])
    heightsquare = (face[0]-face[6])*(face[0]-face[6]) + (face[1]-face[7])*(face[1]-face[7])
    if widthsquare <= 24*24 or heightsquare <= 24*24:
        return 1
    else:
        return 0

def del_small_faces(faces):
    large_faces = []
    for i in range(len(faces)):
        if 0 == isSmallFace(faces[i]) :
            large_faces.append(faces[i])
    return large_faces


def evaluation_single_image(faces_gt, faces_pred, img, ishow):
    
    num_true_positive = 0
    num_false_positive = 0

    trueFaces = []

    has_selected = [0]*len(faces_gt)
    for i in range(len(faces_pred)):
        isFace = 0
        for j in range(len(faces_gt)):
            if has_selected[j]==0:
                isFace = isSameFace(faces_gt[j],faces_pred[i])
                if isFace == 1:
                    has_selected[j] = 1
                    break
        if isFace == 1:
            num_true_positive += 1
        else:
            num_false_positive += 1
        trueFaces.append(isFace)

    if isShow == 1:
        for i in range(len(faces_gt)):
            drawRotateRectange(img,faces_gt[i],(0,0,255),3)
        for i in range(len(faces_pred)):
            if trueFaces[i] == 1:
                drawRotateRectange(img,faces_pred[i],(0,255,0),3)
            else:
                drawRotateRectange(img,faces_pred[i],(255,0,0),3)
            #score = '%3d'%(faces_pred[8])
            #cv2.putText(img, str(faces_pred[8]), (int((faces[0]+faces[4])/2),int((faces[1]+faces[5])/2)), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0 ,0), thickness = 2, lineType = 8)

    num_nodetected_small_faces = 0
    for i in range(len(faces_gt)):
        if has_selected[i] == 0 and  1 == isSmallFace(faces_gt[i]) :
            num_nodetected_small_faces += 1
            
    return [num_true_positive,num_false_positive,num_nodetected_small_faces]
 
if __name__ =='__main__':

    gtfile = 'supervisory_control_data_Rect_2_modified.txt'
    predfile = 'supervisory_control_data_Rect_predict_with_iou03.res'
    srcDir = 'data/'
    dstDir = 'evaluation_Res/'
    notRecallDir = 'det_res_new/notRecall/'
    falseAlarmDir = 'det_res_new/falseAlarm/'
    isShow = 1

    dect_conf_threshold = 0.50
    
    with open(gtfile,'r') as f:
        gt_lines = f.readlines()
    with open(predfile,'r') as f:
        pred_lines = f.readlines()

    num_total_faces = 0
    num_small_faces = 0
    num_true_positive = 0
    num_false_positive = 0

    num_images = 0
    pos_gt = 0
    pos_pred = 0
    while pos_gt < len(gt_lines) and pos_pred < len(pred_lines):
        num_images += 1
        if num_images==100:
            print 'test.....................'
        if 0==num_images % 1000:
            print 'At %s-th image with %s faces, in which we detected %s faces rightly, while %s has been wrong detected as face'%(num_images,(num_total_faces-num_small_faces),num_true_positive,num_false_positive)
        
        name_gt   = gt_lines[pos_gt].strip('\n\t').split()[0]
        name_pred = pred_lines[pos_pred].strip('\n\t').split()[0]

        if not name_gt == name_pred:
            print 'groundtruth name: %s not martch to \n prediction name: %s'%(name_gt,name_pred)
            exit
        pos_gt   += 1
        pos_pred += 1

        img = np.zeros((10,10),np.uint8)
        if 1 == isShow:
            srcPath = srcDir + name_gt + '.jpg'
            img = cv2.imread(srcPath)
        #print pos_gt
        n_gt   = int(gt_lines[pos_gt].strip('\n\t').split()[0])
        n_pred = int(pred_lines[pos_pred].strip('\n\t').split()[0])
        pos_gt   += 1
        pos_pred += 1

        faces_gt   = []
        for i in range(0,n_gt):
            face_gt = gt_lines[pos_gt].strip('\n\t').split()
            pos_gt += 1
            face_gt_array = [float(face_gt[j]) for j in range(len(face_gt)-1)]
            face_gt_array =  change_gt_format(face_gt_array)
            faces_gt.append(face_gt_array)
                    
        faces_pred   = []
        for i in range(0,n_pred):
            face_pred = pred_lines[pos_pred].strip('\n\t').split()
            pos_pred += 1
            #faces_pred.append([float(face_pred[j]) for j in range(len(face_pred))])
            face = [float(face_pred[j]) for j in range(len(face_pred))]
            if( face[len(face)-1] > dect_conf_threshold ):
                faces_pred.append(face)
      
        pred_res = evaluation_single_image(faces_gt, faces_pred, img, isShow)
      
        num_total_faces += n_gt
        num_true_positive += pred_res[0]
        num_false_positive += pred_res[1]
        num_small_faces += pred_res[2]
        
      
        if isShow == 1:
            #cv2.imshow("Test",img)
            #cv2.waitKey(33)
            #savePath = dstDir + name_gt
            #cv2.imwrite(savePath, img)
            
            if not n_gt == pred_res[0] + pred_res[2]:
                wrongPath = notRecallDir + name_gt + '.jpg'
                cv2.imwrite(wrongPath, img)
            if pred_res[1] > 0:
                wrongPath = falseAlarmDir + name_gt + '.jpg'
                cv2.imwrite(wrongPath, img)
                     
            

    print 'There are %s images with %s faces where %s are too small(<24), in which we detected %s faces, while %s has been wrong detected as face'%(num_images,num_total_faces,num_small_faces, num_true_positive,num_false_positive)
    print 'Recall    = %s'%( float(num_true_positive)/(num_total_faces-num_small_faces) )
    print 'Precision = %s'%( float(num_true_positive)/(num_true_positive+num_false_positive))
    print 'FPPI      = %s'%( float(num_false_positive)/num_images )
    
    #cv2.destroyAllWindows()
            

        

	








        
        
