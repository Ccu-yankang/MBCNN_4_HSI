from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from scipy import stats
import numpy as np
import pysaliency
# import warnings

class Evaluate():
    def __init__(self):
        self.report=""
        self.precision = 0
        self.recall = 0
        self.precision_macro = 0
        self.recall_macro = 0
        self.precision_micro = 0
        self.recall_micro = 0
        self.precision_weighted = 0
        self.recall_weighted = 0
        self.auc=0
        self.mae=0
        self.cc=0
        self.nss=0
        self.oters=""
        self.strs=""
        # self.warnings=warnings.filterwarnings('error')      #将警告转化为异常


    def get_precision(self,predict,target):
        return precision_score(target,predict)

    def get_recall(self,predict,target):
        return recall_score(predict,target)

    # def cal_CC(self,predict,target):
    #     map1=target.astype(np.float)
    #     map1=map1-np.mean(map1)
    #     map2=predict.astype(np.float)
    #     map2=map2-np.mean(map2)
    #
    #     cov=np.sum(map1*map2)
    #     d1=np.sum(map1*map1)
    #     d2=np.sum(map2*map2)
    #     cc=cov/(np.sqrt(d1)*np.sqrt(d2))
    #     return cc
    #
    def get_NSS(self,predict,target):
        map=predict.astype(np.float)
        try:
            map=(map-np.mean(map))/np.std(map)
        except RuntimeWarning as e:
            print(np.std(map))
            print(map-np.mean(map))
            print(e)

        nss=np.mean(map[target])
        return nss

    # 计算皮尔逊相关系数
    # predict和target交换顺序不影响
    def get_CC(self,predict,target):
        map1 = target.astype(np.float)
        map1 = map1 - np.mean(map1)
        map2 = predict.astype(np.float) / 255
        map2 = map2 - np.mean(map2)

        cov = np.sum(map1 * map2)
        d1 = np.sum(map1 * map1)
        d2 = np.sum(map2 * map2)
        cc = cov / (np.sqrt(d1) * np.sqrt(d2))
        return cc

    #计算皮尔逊相关系数
    #predict和target交换顺序不影响
    #去他妈的皮尔森系数，维度不对，辣鸡函数
    # def get_CC(self,predict,target):
    #     return stats.pearsonr(predict,target)

    def evaluate(self,predict,target,target_name):
        self.report = classification_report(target, predict, target_names=target_name)
        self.auc = roc_auc_score(target, predict)
        self.mae = mean_absolute_error(target, predict)
        self.precision=precision_score(target,predict,average="binary")
        self.recall=recall_score(target,predict,average="binary")

        self.precision_macro = precision_score(target, predict, average="macro")
        self.recall_macro = recall_score(target, predict, average="macro")
        self.precision_micro = precision_score(target, predict, average="micro")
        self.recall_micro = recall_score(target, predict, average="micro")
        self.precision_weighted = precision_score(target, predict, average="weighted")
        self.recall_weighted = recall_score(target, predict, average="weighted")

    def evaluate_all(self,predict,target,target_name):
        self.report = classification_report(target, predict, target_names=target_name)
        self.auc = roc_auc_score(target, predict)
        self.mae = mean_absolute_error(target, predict)
        self.cc=self.get_CC(predict,target)
        self.nss=self.get_NSS(predict,target)
        self.precision = precision_score(target, predict, average="binary")
        self.recall = recall_score(target, predict, average="binary")

        self.precision_macro = precision_score(target, predict, average="macro")
        self.recall_macro = recall_score(target, predict, average="macro")
        self.precision_micro = precision_score(target, predict, average="micro")
        self.recall_micro = recall_score(target, predict, average="micro")
        self.precision_weighted = precision_score(target, predict, average="weighted")
        self.recall_weighted = recall_score(target, predict, average="weighted")

    def add_others(self,others):
        self.oters=others

    def save_format(self,filename):
        self.strs = "=====================================================\n"+ \
                filename + "\n" +\
                "report:\n" + self.report + "\n" +\
                "precision(average=\"binary\"): "+str(self.precision)+"\n"+\
                "recall (average=\"binary\"): "+str(self.recall)+"\n" +\
                "precision(average=\"micro\"): " + str(self.precision_micro) + "\n" +\
                "recall (average=\"micro\"): " + str(self.recall_micro) + "\n" +\
                "precision(average=\"macro\"): " + str(self.precision_macro) + "\n"+ \
                "recall (average=\"macro\"): " + str(self.recall_macro) + "\n" +\
                "precision(average=\"weighted\"): " + str(self.precision_weighted) + "\n" +\
                "recall (average=\"weighted\"): " + str(self.recall_weighted) + "\n" +\
                "AUC: " + str(self.auc) + "\n" +\
                "MAE: " + str(self.mae) + "\n" +\
                "CC:  " + str(self.cc) +"\n" +\
                "NSS: " + str(self.nss) + "\n" + \
                "others:\n"+ \
                "-----------------------------------------------------\n" +\
                self.oters+"\n"+\
                "=====================================================\n\n"
            # f.write(image_name + '\n')
        # f.write("========================================\n")
        # f.write("report:\n" + report + '\n')
        # f.write("AUC: " + str(auc) + '\n')
        # f.write("MAE: " + str(mae) + '\n\n')

    def save_format_when_trainning(self,filename):
        self.strs = "=====================================================\n" + \
                    filename + "\n" + \
                    "-----------------------------------------------------\n" + \
                    self.oters + "\n" + \
                    "=====================================================\n\n" \

