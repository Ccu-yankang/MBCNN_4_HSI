

def precision(target,ground_truth):

    TP=0
    FP=0
    for i in range(len(target)):
        tp=target[i]*ground_truth[i]        #tp>0:正类预测为正类
        if tp>0:
            TP=TP+1
        elif target[i]>0:
            FP=FP+1
    P=TP/(TP+FP)
    return P

def recall(target,ground_truth):

    TP=0
    FN=0

    for i in range(len(target)):
        tp=target[i]*ground_truth[i]
        fn=target[i]+ground_truth[i]
        if tp>0:
            TP=TP+1
        elif fn==0:
            FN=FN+1
    R=TP/(TP+FN)
    return R

def F_measure(P,R):
    F=2*P*R/(P+R)
    return F