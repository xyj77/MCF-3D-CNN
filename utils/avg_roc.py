#-*- coding:utf-8 -*-
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import roc_curve, auc

def appoint_line(num,file):
    with open(file, 'r') as f:
        i=0
        while i != num-1:
            i = i + 1
            f.readline()
        out = f.readline()
        return out

def plot_avg_roc(path, f_row, t_row, tag = ''):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(10):
        
        file = path + '_' +str(i) + '_roc_record.txt'
        fpr = appoint_line(f_row, file)
        tpr = appoint_line(t_row, file)
        fpr = map(eval, fpr[5:-2].split(", "))
        tpr = map(eval, tpr[5:-2].split(", "))
        # print fpr
        # print tpr
        # raw_input()
        
        # Compute ROC curve and area the curve
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 # label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',
             label='Luck', alpha=.6)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=1.5, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(path[:16]+tag + '_avg_roc.png')
    plt.close('all') # 关闭图 
    print 'Avg ROC curve for %s saved at %s !'%(tag, path[:16])
    