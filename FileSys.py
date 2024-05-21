import os.path
import cv2
import h5py
import fnmatch
import traceback
import torch

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
    def is_file(self,file,mode="first"):
        self.reg=file.split('\\')
        suffix=self.reg[-1].split('.')
        if suffix[0] != suffix[-1] and suffix[-1] in ["mat","txt","jpg","pkl"]:
            if self.reg[0] not in ['.', 'C:', 'c:', 'D:', 'd:', 'E:', 'e:', 'F:', 'f:', 'G:', 'g:', 'H:', 'h:', 'I:', 'i:']:
                self.reg.insert(0, self.current_path)
                if mode=="first":
                    self.current_file = "\\".join(self.reg)
                elif mode=="second":
                    self.current_file="\\".join(self.reg[0:-1])
            else:
                if mode=="first":
                    self.current_file=file
                    self.reg = suffix[-1]
                elif mode=="second":
                    self.current_file="\\".join(self.reg[0:-1])
                    self.reg=self.reg[-1]
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
    def readFile(self, dirs, mode):
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
                data = f.readlines()

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
        elif self.reg == "pkl":
            # model_name = net_name + "_B{}_epoch{}.pkl".format(str(i + 1), str(epoch + 1))
            torch.save(data.state_dict(), self.current_file)
            # torch.save(model[i].state_dict(), os.path.join(net_path, model_name))
            print('save model {}'.format(self.current_file))


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

class Stream(File):
    def __init__(self,net_name,No,epoch,file_name,first_level_path="net",branch_num=1):
        super(Stream, self).__init__()
        self.net_name=net_name
        self.No=No
        self.epoch=epoch
        self.branch_num=branch_num
        self.branch_No=1
        self.file_name=file_name
        self.first_level_path=first_level_path
        self.write_mode="w+"
        self.read_mode="r+"

    def initial(self):
        self.setcwd()
        if not "dataset" in self.current_path_list:
            raise FileNotFoundError("dataset don't exists")
        if not "net" in self.current_path_list:
            os.mkdir(".\\net")
        if not "output" in self.current_path_list:
            os.mkdir(".\\output")
    def dir_name(self):
        if fnmatch.fnmatch(self.first_level_path,"dataset"):
            path = os.path.join(".\\", "dataset")
            path = os.path.join(path, "HS-SOD")
            if fnmatch.fnmatch(self.file_name,"*.mat"):
                path=os.path.join(path,"hyperspectral")
                path=os.path.join(path,self.file_name)
                self.read_mode="r+"
                self.write_mode="w"
            elif fnmatch.fnmatch(self.file_name,"*.jpg"):
                path=os.path.join(path,"ground_truth")
                path=os.path.join(path,self.file_name)
                self.read_mode = cv2.IMREAD_GRAYSCALE
                self.write_mode = "w"
        else:
            path=os.path.join('.\\',self.first_level_path)
            path = os.path.join(path, self.net_name)
            name=self.net_name+"_No.{}".format(self.No)
            path=os.path.join(path,name)
            if fnmatch.fnmatch(self.first_level_path,"net"):
                if self.file_name in ["channel.txt"]:
                    path=os.path.join(path,self.file_name)
                    self.write_mode='a'
                    self.read_mode="r+"
                elif self.file_name in ["information.txt"]:
                    path = os.path.join(path, self.file_name)
                    self.write_mode = 'w+'
                    self.read_mode = "r+"
                elif self.file_name in ["evaluation.txt"]:
                    path=os.path.join(path,"result")
                    path=os.path.join(path,f"epoch{self.epoch}")
                    path=os.path.join(path,self.file_name)
                    self.write_mode = 'a'
                    self.read_mode = "r+"
                elif fnmatch.fnmatch(self.file_name,"*.pkl"):
                    path = os.path.join(path, self.file_name)
                    self.write_mode = 'w'
                    self.read_mode = "r+"
                elif fnmatch.fnmatch(self.file_name,"*.jpg"):
                    path = os.path.join(path, "result")
                    path = os.path.join(path, f"epoch{self.epoch}")
                    path = os.path.join(path, self.file_name)
                    self.write_mode = 'w'
                    self.read_mode = cv2.IMREAD_GRAYSCALE
            elif fnmatch.fnmatch(self.first_level_path,"output"):
                if self.file_name in ["net_info.txt"]:
                    path=os.path.join(path,"net_info")
                    path=os.path.join(path,self.file_name)
                    self.write_mode = 'w'
                    self.read_mode = "r+"
                elif self.file_name in ["evaluation.txt"]:
                    path = os.path.join(path, "net_info")
                    path = os.path.join(path, self.file_name)
                    self.write_mode = 'a'
                    self.read_mode = "r+"
                elif fnmatch.fnmatch(self.file_name,"*jpg"):
                    path=os.path.join(path,"result")
                    path=os.path.join(path,self.file_name)
                    self.write_mode = 'w'
                    self.read_mode = cv2.IMREAD_GRAYSCALE
            elif fnmatch.fnmatch(self.first_level_path,"dataset"):
                pass
        return path

    def make_file(self,dir):
        file=""
        if self.is_file(dir,mode="second"):
            noods=self.current_file
            file=self.reg
        else:
            noods=dir
        noods=noods.split("\\")
        path=""
        for i in noods:
            path=os.path.join(path,i)
            if not os.path.exists(path):
                os.mkdir(path)
        path=os.path.join(path,file)
        if not os.path.exists(path):
            with open(path,'w') as f:
                pass
    def set_readmode(self,mode):
        self.read_mode=mode

    def set_parameters(self,net_name,No,epoch,file_name,first_level_path,branch_No=1):
        self.net_name = net_name
        self.No = No
        self.epoch = epoch
        self.file_name = file_name
        self.first_level_path = first_level_path
        # **************************************************************************************
        self.branch_No=branch_No
        # **************************************************************************************

    def scan_imageList(self):
        path=".\\dataset\\HS-SOD\\hyperspectral\\"
        for root,path,file in os.walk(path):
            return file

    def write(self,data):
        try:
            dir = self.dir_name()
            self.writeFile(dir,data,self.write_mode)
        except FileNotFoundError as e:
            print(e)
            print(traceback.format_exc())
            print("create dir {}".format(dir))
            self.make_file(dir)
            self.writeFile(dir, data, self.write_mode)
            print("create dir {} successfully".format(dir))
        except PermissionError as e:
            print(e)
            print(traceback.format_exc())
        except Exception as e:
            print(e)

    def read(self):
        try:
            dir=self.dir_name()
            return self.readFile(dir,self.read_mode)
        except FileNotFoundError as e:
            print(e)
            print(traceback.format_exc())
            print("create dir {}".format(dir))
            self.make_file(dir)
            print("create dir {} successfully".format(dir))
            self.readFile(dir, self.read_mode)
        except PermissionError as e:
            print(e)
            print(traceback.format_exc())