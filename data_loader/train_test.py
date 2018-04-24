# -*- coding:utf-8 -*-
import os, sys
root = os.getcwd()
test_list = ['0132780021', '0217034311', '0313618411', '0009096012', '0015925312',\
             '0272042222', '0337448822', '0392973812', '0416039612', '0488100312',\
             '0489631412', '0814982912', '0019041513', '0019041533', '0415657313']

             
def file_del(path):
    for img in os.listdir(path):
        if (str(img[:10]) in ['0043162012', '1321267612']) & (int(path[-1]) == 0):
            os.remove(os.path.join(path, img))
            print ("    Delete File: " + os.path.join(path, img))        
                
def split_dir(dir):
    fp_train = open(os.path.join(dir, "train.txt"), 'w')
    fp_test = open(os.path.join(dir, "test.txt"), 'w')
    for sub_dir in os.listdir(dir):
        path = os.path.join(dir, sub_dir)
        if os.path.isdir(path):
            if int(sub_dir) == 0:
                file_del(path)
            for img in os.listdir(path):
                Level = sub_dir
                p = os.path.join(path, img)
                p = os.path.join(root, p)
                if str(img[:10]) in test_list:
                    fp_test.write(Level + ' ' + p + '\n')
                else:
                    fp_train.write(Level + ' ' + p + '\n')
    fp_test.close()
    fp_train.close()

def main(argv=sys.argv):
    # dir = argv[1]
    # write_dir(dir = dir)
    # 批量获取文件路径
    for dir in os.listdir(root):
        path = os.path.join(root, dir)
        # print os.path.isdir(path)
        if os.path.isdir(path):
            split_dir(dir = path)
            print path + '  done!'

if __name__ == '__main__':
    main(sys.argv)
