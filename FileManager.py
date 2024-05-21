import os.path

import cv2
import h5py

class Names():
    def __init__(self,net="End_to_End",layer=5,branch_num=1,No=1,trainning=True):
        self.net=net                #网络
        self.layer=layer            #网络层数
        self.branch_num=branch_num  #分支数
        self.No=No                  #网络编号
        self.trainning = trainning  # 是否是训练网络
        self.net_name=self.get_name()

        self.net_path = 'net'
        self.output_path = 'output'

        self.first_net_path = ""
        self.first_output_path = ""

        self.second_net_path = ""
        self.second_output_path = ""

        self.saliency_output_path = ""
        self.resultInfo_output_path = ""

        self.dataset_path = 'dataset'
        self.dataset_HSSOD_path = 'HS-SOD'
        # self.dataset_hyperspectral_path='hyperspectral'
        self.dataset_hyperspectral_path = ".\\dataset\\HS-SOD\\hyperspectral\\"
        self.dataset_ground_truth_path = ".\\dataset\\HS-SOD\\ground_truth\\"

        self.evaluationFile = 'evaluation.txt'
        self.netInfoFile = 'info.txt'
        self.channelFile = 'channel.txt'
        self.set_listFile = 'information.txt'
        self.No = 1
        self.branch_num = 1
        self.sample_channels = 54
        self.model_flag = False
        self.channel_list=[]

    def set_Name(self,net,layer,branch_num,No,trainning):
        self.net=net
        self.layer=layer
        self.branch_num=branch_num
        self.No=No
        self.trainning=trainning

    def set_No(self,No):
        self.No=No

    def add_No(self):
        self.No+=1

    def get_name(self):
        self.net_name = self.net + f"_L{self.layer}" + f"_BN{self.branch_num}"
        return self.net_name

    def get_netName(self):
        if self.trainning:
            self.net_name=self.net+f"_L{self.layer}"+f"_BN{self.branch_num}"+f"_No.{self.No}"
        else:
            self.net_name="TestNet_"+self.net+f"_NN{self.branch_num}"+f"_No.{self.No}"
        return self.net_name

class File():
    def __init__(self):
        self.dirs = {}                                              # 文件路径
        self.paths = {}                                             # 文件夹路径
        self.tree = []                                              # 文件树
        self.root_path = os.getcwd()                                # 文件系统根目录
        self.current_path = self.root_path                          # 当前文件系统工作目录路径
        self.current_path_list = []                                 # 当前文件系统工作目录文件夹列表
        self.current_file_list = []                                 # 当前文件系统工作目录文件列表
        self.current_file=[]                                        # 当前文件路径
        self.image_train_list = []
        self.image_test_list = []
        self.reg=[]                                                 # 临时变量
        self.setcwd()

    # 添加文件路径
    def add_dir(self, dirs, names):
        if isinstance(dirs,list):
            i = 0
            for n in dirs:
                self.dirs[names[i]] = n
                i += 1
        else:
            self.dirs[names]=dirs

    # 删除文件路径
    def del_dir(self, names):
        if isinstance(names, list):
            i = 0
            for n in names:
                del self.dirs[names[i]]
                i += 1
        else:
            del self.dirs[names]

    # 添加文件夹路径
    def add_path(self, paths, names):
        if isinstance(paths, list):
            i = 0
            for n in paths:
                self.paths[names[i]] = n
                i += 1
        else:
            self.paths[names]=paths

    # 删除文件夹路径
    def del_dir(self, names):
        if isinstance(names, list):
            i = 0
            for n in names:
                del self.paths[names[i]]
                i += 1
        else:
            del self.paths[names]

    #判断是否是绝对路径，并将相对路径转换为绝对路径存储再self.current_file中
    def is_absolutePath(self,paths):
        self.reg=paths.split("\\")
        if self.reg[0] not in ['.', 'C:', 'c:', 'D:', 'd:', 'E:', 'e:', 'F:', 'f:', 'G:', 'g:', 'H:', 'h:', 'I:', 'i:']:
            self.reg.insert(0,self.current_path)
            self.current_file='\\'.join(self.reg)+'\\'
            return False
        else:
            self.current_file=self.reg
            return True

    # 判断是否是相对路径，并将相对路径转换为绝对路径存储再self.current_file中
    def is_relativePath(self, paths):
        self.reg = paths.split("\\")
        if self.reg[0] not in ['.', 'C:', 'c:', 'D:', 'd:', 'E:', 'e:', 'F:', 'f:', 'G:', 'g:', 'H:', 'h:', 'I:', 'i:']:
            self.reg.insert(0, self.current_path)
            self.current_file = '\\'.join(self.reg)
            return True
        else:
            self.current_file = paths
            return False

    #判断路径是否为指定格式文件（.mat;.txt;.jpg），并将相对路径转换为绝对路径存储再self.current_file中，将文件后缀存储再self.reg中
    def is_file(self,file):
        self.reg=file.split('\\')
        suffix=self.reg[-1].split('.')
        if suffix[0] != suffix[-1] and suffix[-1] in ["mat","txt","jpg"]:
            if self.reg[0] not in ['.', 'C:', 'c:', 'D:', 'd:', 'E:', 'e:', 'F:', 'f:', 'G:', 'g:', 'H:', 'h:', 'I:', 'i:']:
                self.reg.insert(0, self.current_path)
                self.current_file = '\\'.join(self.reg)
            self.reg=suffix[-1]
            return True
        else:
            return False

    # 创建文件夹
    def create_path(self, paths):
        if isinstance(paths,list):
            for n in paths:
                self.is_relativePath(n)
                if not os.path.exists(self.current_file):
                    os.makedirs(self.current_file,exist_ok=True)
        else:
            self.is_relativePath(paths)
            if not os.path.exists(self.current_file):
                os.makedirs(self.current_file,exist_ok=True)
    #创建文件
    def create_file(self,files):
        if isinstance(files,list):
            for n in files:
                self.is_file(n)
                with open(self.current_file,'w') as f:
                    pass
        else:
            self.is_file(files)
            with open(self.current_file, 'w') as f:
                pass

    #设置当前文件系统工作路径
    def setcwd(self,path=os.getcwd()):
        self.is_relativePath(path)
        self.current_path=self.current_file
        # self.current_path_list=os.listdir(self.current_path)
        for root,path,file in os.walk(self.current_path):
            self.current_path_list=path
            self.current_file_list=file
            break

    #返回上级目录
    def parent_dir(self):
        self.reg=self.current_path.split('\\')
        self.current_path='\\'.join(self.reg[0:-1])
        return self.current_path

    # 获取当前文件系统工作路文件夹径列表
    def get_pathlist(self):
        return self.current_path_list

    # 获取当前文件系统工作路径文件列表
    def get_filelist(self):
        return self.current_file_list

    # 读取文件
    # mode={"r+",cv2.IMREAD_GRAYSCALE,cv2.IMREAD_COLOR}
    # 返回值：
    # False:读取失败
    def readFile(self, dirs, mode, lines=1):
        if not self.is_file(dirs):
            return False
        # 读取mat文件
        if self.reg == 'mat':
            data = h5py.File(self.current_file)
        #读取jpg文件
        elif self.reg == 'jpg':
            if mode not in [cv2.IMREAD_GRAYSCALE,cv2.IMREAD_COLOR]:
                return False
            data = cv2.imread(self.current_file, mode)
        #读取txt文件
        elif self.reg == 'txt':
            # mode默认等于"r+"
            if mode not in ['r','r+']:
                mode = 'r+'
            with open(self.current_file, mode) as f:
                data = f.readlines(lines)

        return data

    # 写入文件
    # mode={"w+","r+","a"}
    # 返回值：
    # False:写入失败
    def writeFile(self, dirs, data, mode):
        if not self.is_file(dirs):
            return False
        # 写入mat文件
        if self.reg == 'mat':
            pass
        elif self.reg == 'jpg':
            cv2.imwrite(self.current_file, data)
        elif self.reg == 'txt':
            # mode默认等于"w+"
            if mode not in ['w','w+','a']:
                mode='w+'
            with open(self.current_file, mode) as f:
                f.write(data)

    #参数：
    #path:保存路径
    def create_SetList(self,setlist,path):
        setlist.sort()
        self.image_train_list = setlist[0::2]
        self.image_test_list = setlist[1::2]
        data = setlist[0::2]
        data.insert(0, "train_list:")
        data.append("test_list:")
        data.extend(self.image_test_list)
        self.setcwd(path)
        self.create_file('information.txt')
        self.writeFile('information.txt', "\n".join(data), "w+")
