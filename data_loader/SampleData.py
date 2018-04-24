#-*- coding:utf -8-*-
#http://www.cnblogs.com/huadongw/p/6159408.html
#数据重采样
#python SampleData.py -s 0 trainJ/train.txt 64 trainJ/trainSample.txt
#
# 从python调用shell脚本
# !/usr/bin/python
# import sys
# import os
# print "start call sh file"
# os.system('./fromsh.sh')
# print "end call sh file"
#
# 从shell脚本调用python
# !/bin/bash
# echo 'start call py'
# ./frompy.py
# echo 'end call py'
#
#
import numpy as np
from sklearn.utils import check_random_state
import os, sys, math, random
from collections import defaultdict
if sys.version_info[0] >= 3:
    xrange = range
 
def exit_with_help(argv):
    print("""\
Usage: {0} [options] dataset subclass_size [output]
options:
-s method : method of selection (default 0)
     0 -- over-sampling & under-sampling given subclass_size
     1 -- over-sampling (subclass_size: any value)
     2 -- under-sampling(subclass_size: any value)
 
output : balance set file (optional)
If output is omitted, the subset will be printed on the screen.""".format(argv[0]))
    exit(1)
 
def process_options(argv):
    argc = len(argv)
    if argc < 3:
        exit_with_help(argv)
 
    # default method is over-sampling & under-sampling
    method = 0 
    BalanceSet_file = sys.stdout
 
    i = 1
    while i < argc:
        if argv[i][0] != "-":
            break
        if argv[i] == "-s":
            i = i + 1
            method = int(argv[i])
            if method not in [0,1,2]:
                print("Unknown selection method {0}".format(method))
                exit_with_help(argv)
        i = i + 1
 
    dataset = argv[i] 
    BalanceSet_size = int(argv[i+1])
 
    if i+2 < argc:
        BalanceSet_file = open(argv[i+2],'w')
 
    return dataset, BalanceSet_size, method, BalanceSet_file
 
def stratified_selection(dataset, subset_size, method):
    labels = [line.split(None,1)[0] for line in open(dataset)]
    label_linenums = defaultdict(list)
    for i, label in enumerate(labels):
        label_linenums[label] += [i]
 
    l = len(labels)
    remaining = subset_size
    ret = []
 
    # classes with fewer data are sampled first;
    label_list = sorted(label_linenums, key=lambda x: len(label_linenums[x]))
    min_class = label_list[0]
    maj_class = label_list[-1]
    min_class_num = len(label_linenums[min_class])
    maj_class_num = len(label_linenums[maj_class])
    random_state = check_random_state(42)
 
    for label in label_list:
        linenums = label_linenums[label]
        label_size = len(linenums)
        if  method == 0:
            if label_size<subset_size:
                ret += linenums
                subnum = subset_size-label_size
            else:
                subnum = subset_size
            ret += [linenums[i] for i in random_state.randint(low=0, high=label_size,size=subnum)]
        elif method == 1:
            if label == maj_class:
                ret += linenums
                continue
            else:
                ret += linenums
                subnum = maj_class_num-label_size               
                ret += [linenums[i] for i in random_state.randint(low=0, high=label_size,size=subnum)]
        elif method == 2:
            if label == min_class:
                ret += linenums
                continue
            else:
                subnum = min_class_num
                ret += [linenums[i] for i in random_state.randint(low=0, high=label_size,size=subnum)]
    random.shuffle(ret)
    return ret

def sampledata(dataset, subset_size, method, subset):
    selected_lines = []
    selected_lines = stratified_selection(dataset, subset_size,method)
 
    #select instances based on selected_lines
    subset_file = open(subset,'w')
    dataset = open(dataset,'r')
    datalist = dataset.readlines()
    for i in selected_lines:
        subset_file.write(datalist[i])
    
    subset_file.close()
    dataset.close()
   
def main(argv=sys.argv):
    dataset, subset_size, method, subset_file = process_options(argv)
    selected_lines = []
 
    selected_lines = stratified_selection(dataset, subset_size,method)
 
    #select instances based on selected_lines
    dataset = open(dataset,'r')
    datalist = dataset.readlines()
    for i in selected_lines:
        subset_file.write(datalist[i])
    subset_file.close()
 
    dataset.close()
 
if __name__ == '__main__':
    main(sys.argv)