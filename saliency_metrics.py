import random
import FileSys as fs
import Process as process
import net
import Evaluate as eva
import fnmatch
import os
import re
import cv2

class evaluate_cell():
    def __init__(self):
        self.name=""
        self.report=[]
        self.precision_binary=""
        self.recall_binary=""
        self.precision_micro=""
        self.recall_micro=""
        self.precision_macro=""
        self.recall_macro=""
        self.precision_weighted=""
        self.recall_weighted=""
        self.AUC=""
        self.MAE=""
        self.CC=""
        self.NSS=""
        self.others=[]
        self.others_str=""
        self.current_variable=None
        self.completed=False                #数据是否完全

class evaluate_file():
    def __init__(self):
        self.name_list=[]
        self.need_to_complete=[]            #需要补全评价指标的图片列表
        self.cells=[]
        self.len=0
        self.start_of_cell=False
        self.line_point=0
        self.eva=eva.Evaluate()
        self.process=process.process()

    def add_cell(self,cell):
        self.cells.append(cell)
        self.len+=1

    def pop_cell(self):
        if self.len>0:
            self.len -= 1
            return self.cells.pop()

    def process_data(self,data):
        strs=data.split("\n")
        count=0
        for line in strs:
            # 出现多次=号，cell结束或开始()
            if re.match("[=]{3,}",line):            #[=]{3,}:中括号匹配字符范围，大括号匹配的最低次数和最大次数，最大次数省略
                if not self.start_of_cell:              #读取cell
                    self.start_of_cell=True
                    self.cells.append(evaluate_cell())
                    self.len+=1
                    count = 0
                else:
                    if count==15:
                        self.cells[-1].completed=False
                        self.need_to_complete.append(self.cells[-1].name)
                        count=0
                    else:
                        self.cells[-1].completed=False
                        self.need_to_complete.append(self.cells[-1].name)
                    self.start_of_cell=False            #结束cell
            elif fnmatch.fnmatch(line,"*.mat"):         #可以使用更精准的匹配规则
                self.name_list.append(line)
                self.cells[-1].name=line
                count+=1
            elif fnmatch.fnmatch(line,"report*"):
                self.cells[-1].current_variable="report"
                count += 1
            elif fnmatch.fnmatch(line,"precision(average=\"binary\")*"):
                self.cells[-1].precision_binary=line.split(":")[-1]
                count += 1
            elif fnmatch.fnmatch(line,"recall (average=\"binary\")*"):
                self.cells[-1].recall_binary=line.split(":")[-1]
                count += 1
            elif fnmatch.fnmatch(line,"precision(average=\"micro\")*"):
                self.cells[-1].precision_micro=line.split(":")[-1]
                count += 1
            elif fnmatch.fnmatch(line,"recall (average=\"micro\")*"):
                self.cells[-1].recall_micro=line.split(":")[-1]
                count += 1
            elif fnmatch.fnmatch(line,"precision(average=\"macro\")*"):
                self.cells[-1].precision_macro=line.split(":")[-1]
                count += 1
            elif fnmatch.fnmatch(line,"recall (average=\"macro\")*"):
                self.cells[-1].recall_macro=line.split(":")[-1]
                count += 1
            elif fnmatch.fnmatch(line,"precision(average=\"weighted\")*"):
                self.cells[-1].precision_weighted=line.split(":")[-1]
                count += 1
            elif fnmatch.fnmatch(line,"recall (average=\"weighted\")*"):
                self.cells[-1].recall_weighted=line.split(":")[-1]
                count += 1
            elif fnmatch.fnmatch(line,"AUC*"):
                self.cells[-1].AUC=line.split(":")[-1]
                count += 1
            elif fnmatch.fnmatch(line,"MAE*"):
                self.cells[-1].MAE=line.split(":")[-1]
                count += 1
            elif fnmatch.fnmatch(line,"CC*"):
                self.cells[-1].CC=line.split(":")[-1]
                count += 1
            elif fnmatch.fnmatch(line,"NSS*"):
                self.cells[-1].NSS=line.split(":")[-1]
                count += 1
            elif fnmatch.fnmatch(line,"others*"):
                self.cells[-1].current_variable="others"
                count += 1
            elif re.match("[-]{3,}", line):
                pass
            elif fnmatch.fnmatch(line,""):
                pass
            elif self.cells[-1].current_variable is not None:
                if self.cells[-1].current_variable == "report":

                    pass                                                    #暂不处理
                elif self.cells[-1].current_variable == "others":
                    self.cells[-1].others_str=self.cells[-1].others_str+line+"\n"
                    item=line.split("=")
                    self.cells[-1].others.append({item[0]:item[1]})
                    # count += 1

    def evaluate_img(self,data,gt):
        self.eva.evaluate_all(data,gt,["class0","class1"])
        cell=evaluate_cell()
        cell.report = self.eva.report
        cell.precision_binary = self.eva.precision
        cell.recall_binary = self.eva.recall
        cell.precision_micro = self.eva.precision_micro
        cell.recall_micro = self.eva.recall_micro
        cell.precision_macro = self.eva.precision_macro
        cell.recall_macro = self.eva.recall_macro
        cell.precision_weighted = self.eva.precision_weighted
        cell.recall_weighted = self.eva.recall_weighted
        cell.AUC = self.eva.auc
        cell.MAE = self.eva.mae
        cell.CC = self.eva.cc
        cell.NSS = self.eva.nss
        cell.completed=True
        return cell


    #将cell转换成字符串，方便保存
    def cell2str(self,cell):
        if cell.others_str[-1] == "\n":
            others_end=""
        else:
            others_end="\n"
        strs = "=====================================================\n" + \
                    cell.name + "\n" + \
                    "report:\n" + cell.report + "\n" + \
                    "precision(average=\"binary\"): " + str(cell.precision_binary) + "\n" + \
                    "recall (average=\"binary\"): " + str(cell.recall_binary) + "\n" + \
                    "precision(average=\"micro\"): " + str(cell.precision_micro) + "\n" + \
                    "recall (average=\"micro\"): " + str(cell.recall_micro) + "\n" + \
                    "precision(average=\"macro\"): " + str(cell.precision_macro) + "\n" + \
                    "recall (average=\"macro\"): " + str(cell.recall_macro) + "\n" + \
                    "precision(average=\"weighted\"): " + str(cell.precision_weighted) + "\n" + \
                    "recall (average=\"weighted\"): " + str(cell.recall_weighted) + "\n" + \
                    "AUC: " + str(cell.AUC) + "\n" + \
                    "MAE: " + str(cell.MAE) + "\n" + \
                    "CC:  " + str(cell.CC) + "\n" + \
                    "NSS: " + str(cell.NSS) + "\n" + \
                    "others:\n" + \
                    "-----------------------------------------------------\n" + \
                    cell.others_str + others_end + \
                    "=====================================================\n\n"
        return strs

    #将cell列表转换成字符串,方便保存
    def cells2str(self):
        strs=""
        for cell in self.cells:
            strs=strs+self.cell2str(cell)
        return strs

#=================================
if __name__=='__main__':
    location="End_to_End_L5_BN1"
    net_path=".\\net"
    temp_path = ".\\net\\temp"
    path_level=1
    first_level_path_list=None
    second_level_path_list=None
    third_level_path_list=None
    fouth_level_path_list=None
    #更改目录
    current_path=net_path
    stream=fs.Stream(net_name=location,No=1,epoch=1,file_name="0001.jpg",first_level_path="dataset")
    stream.initial()
    if location is not None:
        for root,paths,files in os.walk(current_path):
            #可以指定目录层级缩短搜索时间
            have_img = False
            have_evaluate = False
            #当前路径下是否存在图片和evaluation.txt
            if "evaluation.txt" in files:
                have_evaluate=True
                for file in files:
                    if fnmatch.fnmatch(file,"*.jpg"):
                        have_img=True
                        break

            #没有图片，继续搜索
            if not have_img:
                continue

            #有图片，继续处理
            evafile = evaluate_file()
            with open(os.path.join(root,"evaluation.txt"),'r') as f:
                print("open file at {}".format(os.path.join(root,"evaluation.txt")))
                strs=f.read()
            evafile.process_data(strs)
            if evafile.need_to_complete.__len__()>0:
                for item in evafile.need_to_complete:
                    jpg_name=item.replace(".mat",".jpg")
                    if jpg_name in files:
                        #读取图像
                        dir=os.path.join(root,jpg_name)
                        #读取真值图
                        img=stream.readFile(dir,cv2.IMREAD_GRAYSCALE)
                        net_name = root.split("\\")
                        net_name = net_name[1]
                        stream.set_parameters(net_name=net_name, No=1, epoch=1, file_name=jpg_name,
                                              first_level_path="dataset")
                        ground_truth = stream.read()
                        #计算评价指标
                        ground_truth=evafile.process.ground_truth_process(ground_truth)
                        img=evafile.process.img_process_with_jpg(img)
                        img ,ground_truth=evafile.process.eva_process_with_jpg(img,ground_truth)
                        cell=evafile.evaluate_img(img,ground_truth)
                        index = evafile.name_list.index(item)
                        cell.name=item
                        cell.others_str=evafile.cells[index].others_str
                        cell.others=evafile.cells[index].others
                        #替换cell列表中的评价指标
                        evafile.cells[index]=cell
                        print("metrics calculated down at {}".format(jpg_name))
                    else:
                        continue
                try:
                    strs = evafile.cells2str()
                except IndexError as e:
                    print(e)
                    print("Error at "+item)
                    raise
                with open(os.path.join(root, "evaluation.txt"), 'w+') as f:
                    f.write(strs)

                print("search done at dir : {}".format(root))


