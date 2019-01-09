from __future__ import division
from sklearn import metrics

def accuracy(predict, truth):
    return metrics.accuracy_score(truth,predict)
    #correct = [index for index, value in enumerate(predict) if value == truth[index]]
    #print correct
    #return len(correct)/len(truth)

def mutual_info(predict, truth):
    return metrics.normalized_mutual_info_score(truth, predict)
    #return metrics.average_precision_score(truth, predict)