import pprint,cPickle
import numpy as np
# from scipy.linalg.misc import norm

import datetime
starttime = datetime.datetime.now()
pkl_file = open('./name_array.pkl', 'rb')
name_array = cPickle.load(pkl_file)
pkl_file.close()
endtime = datetime.datetime.now()
print (endtime - starttime).seconds


pkl_file1 = open('./name_start_end.pkl', 'rb')
name_start_end = cPickle.load(pkl_file1)
pkl_file1.close()

def find_index(pair1_1,pair1_2):
    pair_all = [pair1_1,pair1_2]
    if pair_all in tmp_name_start_end:
        index_loc = tmp_name_start_end.index(pair_all)
    start = name_start_end[index_loc][2]
    end = name_start_end[index_loc][3]
    return (start,end)

def ave_pool(start,end) :
    tmp_feat = []
    len_para = end - start + 1
    for i in range(start,end):
        tmp_feat.append(name_array[i][4])
    feat_128 = map(sum, zip(*tmp_feat))
    feat_128 = [x * (1.0/len_para) for x in feat_128]
    return feat_128

def max_pool(start,end) :
    tmp_feat = []
    len_para = end - start + 1
    for i in range(start,end):
        tmp_feat.append(name_array[i][4])
    feat_128 = map(max, zip(*tmp_feat))
    feat_128 = [x * (1.0/len_para) for x in feat_128]
    return feat_128

def mean_l2(start1,end1,start2,end2):
    num1 = end1 - start1 + 1
    num2 = end2 - start2 + 1
    dist_all = [[] for i in range(num1*num2)]
    tmp_count = 0
    for i in range(num1):
        for j in range(num2):
            feat1 = name_array[start1+i][4]
            feat2 = name_array[start2+j][4]
            feat = list(map(lambda x: x[0] - x[1], zip(feat1, feat2)))
            feat = np.array(feat)
            dist = np.linalg.norm(feat, ord=None)
            dist_all[tmp_count] = dist
            tmp_count += 1
    mean_dist = np.mean(dist_all)
    return mean_dist

def min_l2(start1,end1,start2,end2):
    num1 = end1 - start1 + 1
    num2 = end2 - start2 + 1
    dist_all = [[] for i in range(num1*num2)]
    tmp_count = 0
    for i in range(num1):
        for j in range(num2):
            feat1 = name_array[start1+i][4]
            feat2 = name_array[start2+j][4]
            feat = list(map(lambda x: x[0] - x[1], zip(feat1, feat2)))
            feat = np.array(feat)
            dist = np.linalg.norm(feat, ord=None)
            dist_all[tmp_count] = dist
            tmp_count += 1
    min_dist = sorted(dist_all, key=lambda dist_orig: dist_orig[0])[0]
    return min_dist

def cal_dist(ave_feat1,ave_feat2):
    minus_feat = list(map(lambda x: x[0]-x[1], zip(ave_feat1, ave_feat2)))
    minus_feat = np.array(minus_feat)
    dist = np.linalg.norm(minus_feat, ord=None)
    return dist

tmp_count = 0
dist_label = []
tmp_dist_label = [[],'']
start_line = 1
folders_num = len(name_start_end)
tmp_name_start_end = [['',''] for i in range(folders_num)]
for i in range(folders_num):
    tmp_name_start_end[i][0] = name_start_end[i][0]
    tmp_name_start_end[i][1] = name_start_end[i][1]

for line in open("split_ytf.txt"):
    if tmp_count == 0:
        tmp_count += 1
        continue
    if tmp_count in range(1,501):
        keys = line.split(',')
        pair1 = keys[2]
        pair2 = keys[3]
        whet_same = keys[4][0]
        pair1_1 = pair1[0:-2]
        pair1_2 = pair1[-1]
        pair2_1 = pair2[0:-2]
        pair2_2 = pair2[-1]
        (start1,end1) = find_index(pair1_1,pair1_2)
        (start2,end2) = find_index(pair2_1,pair2_2)
        # ave_feat1 = ave_pool(start1,end1)
        # ave_feat2 = ave_pool(start2,end2)
        # dist = cal_dist(ave_feat1,ave_feat2)
        dist = mean_l2(start1,end1,start2,end2)
        tmp_dist_label = [[], '']
        tmp_dist_label[0] = dist
        tmp_dist_label[1] = whet_same
        dist_label.append(tmp_dist_label)
        tmp_count += 1
    else:
        break

output2 = open('dist_label.pkl','wb')
cPickle.dump(dist_label,output2)
output2.close()

aa = 1

