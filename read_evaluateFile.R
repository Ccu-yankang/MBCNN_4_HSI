#网址:
#https://zhidao.baidu.com/question/1928330416005685547.html
# file="D:/workspace/python_workspace/MBCNN/MBCNN/net/End_to_End_L5_BN1/End_to_End_L5_BN1_No.1/result/epoch1/evaluation.txt"

#读取evaluation.txt文件并返回数据框
read_file=function(file)
{
  # print(file)
  file_str=readLines(file)
  
  #创建空数据框
  df=data.frame()
  # colnames(df)=c('img_name','precision','recall','AUC','MAE','CC','NSS')
  start_of_cell=FALSE
  current_variable="None"
  img_name=""
  precision_binary=0
  recall_binary=0
  precision_micro=0
  recall_micro=0
  precision_macro=0
  recall_macro=0
  precision_weighted=0
  recall_weighted=0
  AUC=0
  MAE=0
  CC=0
  NSS=0
  loss_crossentropy=0
  loss_precision=0
  acos_loss_precision=0
  loss_recall=0
  acos_loss_recall=0
  loss_fn=0
  net_name="default_net"
  net_epoch=0
  for(line in file_str){
    if (grepl("[=]{3,}",line,ignore.case=FALSE)){
      if (!start_of_cell){
        start_of_cell=TRUE
      }else{
        vector=c(net_name,img_name,net_epoch,AUC,MAE,CC,NSS,precision_binary,recall_binary,precision_macro,recall_macro,
                 precision_micro,recall_micro,precision_weighted,recall_weighted,loss_crossentropy,
                 loss_precision,acos_loss_precision,loss_recall,acos_loss_recall,loss_fn)
        df=rbind(df,vector)
        start_of_cell=FALSE
      }
    }else if(grepl(".*.mat",line)){
      img_name=line
    }else if(grepl("report.*",line)){
      current_variable="report"
    }else if(grepl("precision.*binary.*",line)){
      # table=read.table(textConnection(object = line),header = FALSE,sep=":",dec='.')
      # precision_binary=table[1,2]
      strs=unlist(strsplit(line,":"))
      precision_binary=format(strs[2],nsmall = 17)             #nsmall可以大一点，小于20都行
    }else if(grepl("recall.*binary.*",line)){
      strs=unlist(strsplit(line,":"))
      recall_binary=format(strs[2],nsmall = 17)
    }else if(grepl("precision.*micro*",line)){
      strs=unlist(strsplit(line,":"))
      precision_micro=format(strs[2],nsmall=17)
    }else if(grepl("recall.*micro*",line)){
      strs=unlist(strsplit(line,":"))
      recall_micro=format(strs[2],nsmall = 17)
    }else if(grepl("precision.*macro*",line)){
      strs=unlist(strsplit(line,":"))
      precision_macro=format(strs[2],nsmall = 17)
    }else if(grepl("recall.*macro*",line)){
      strs=unlist(strsplit(line,":"))
      recall_macro=format(strs[2],nsmall = 17)
    }else if(grepl("precision.*weighted*",line)){
      strs=unlist(strsplit(line,":"))
      precision_weighted=format(strs[2],nsmall = 17)
    }else if(grepl("recall.*weighted*",line)){
      strs=unlist(strsplit(line,":"))
      recall_weighted=format(strs[2],nsmall = 17)
    }else if(grepl("AUC.*",line)){
      strs=unlist(strsplit(line,":"))
      AUC=format(strs[2],nsmall = 17)
    }else if(grepl("MAE.*",line)){
      strs=unlist(strsplit(line,":"))
      MAE=format(strs[2],nsmall = 17)
    }else if(grepl("CC.*",line)){
      strs=unlist(strsplit(line,":"))
      CC=format(strs[2],nsmall = 17)
    }else if(grepl("NSS.*",line)){
      strs=unlist(strsplit(line,":"))
      NSS=format(strs[2],nsmall = 17)
    }else if(grepl(".*loss_crossentropy",line)){
      strs=unlist(strsplit(line,"[(]"))     #标点符号作为分隔符这样用，详见:help("strsplit")
      strs=unlist(strsplit(strs[2],","))
      loss_crossentropy=strs[1]
    }else if(grepl("branch1:loss_precision",line)){
      strs=unlist(strsplit(line,"[(]"))     #标点符号作为分隔符这样用，详见:help("strsplit")
      strs=unlist(strsplit(strs[2],"[)]"))
      loss_precision=strs[1]
    }else if(grepl("branch1:acos_loss_precision",line)){
      strs=unlist(strsplit(line,"[(]"))     #标点符号作为分隔符这样用，详见:help("strsplit")
      strs=unlist(strsplit(strs[2],"[)]"))
      acos_loss_precision=strs[1]
    }else if(grepl("branch1:loss_recall",line)){
      strs=unlist(strsplit(line,"[(]"))     #标点符号作为分隔符这样用，详见:help("strsplit")
      strs=unlist(strsplit(strs[2],"[)]"))
      loss_recall=strs[1]
    }else if(grepl("branch1:acos_loss_recall",line)){
      strs=unlist(strsplit(line,"[(]"))     #标点符号作为分隔符这样用，详见:help("strsplit")
      strs=unlist(strsplit(strs[2],"[)]"))
      acos_loss_recall=strs[1]
    }else if(grepl("branch1:loss_fn",line)){
      strs=unlist(strsplit(line,"[(]"))     #标点符号作为分隔符这样用，详见:help("strsplit")
      strs=unlist(strsplit(strs[2],","))
      loss_fn=strs[1]
    }else if(grepl("others.*",line)){
      current_variable="others"
    }else if(grepl("[-]{3,}",line)){
    }else if(grepl("",line)){
    }else if(current_variable != "None"){
      if(current_variable == "report"){
        #暂不处理
      }else if(current_variable == "others"){
        #不在这里做处理
      }
    }
    
  }
  colnames(df)=c('net_name','img_name','net_epoch','AUC','MAE','CC','NSS','precision_binary','recall_binary','precision_macro','recall_macro',
                 'precision_micro','recall_micro','precision_weighted','recall_weighted','loss_crossentropy',
                 'loss_precision','acos_loss_precision','loss_recall','acos_loss_recall','loss_fn')
  return(df)
}

#读取evaluation.txt文件并返回数据框，数据框中的数据为数值型
read_file_with_numeric=function(file)
{
  # print(file)
  file_str=readLines(file)
  
  #创建空数据框
  df=data.frame()
  start_of_cell=FALSE
  current_variable="None"
  default_net_name="default_net"
  for(line in file_str){
    if (grepl("[=]{3,}",line,ignore.case=FALSE)){
      if (!start_of_cell){
        dv=data.frame(
          net_name=default_net_name,img_name="",net_epoch=0L,
          AUC=0.0,MAE=0.0,CC=0.0,NSS=0.0,precision_binary=0.0,
          recall_binary=0.0,precision_macro=0.0,recall_macro=0.0,
          precision_micro=0.0,recall_micro=0.0,precision_weighted=0.0,
          recall_weighted=0.0,loss_crossentropy=0.0,loss_precision=0.0,
          acos_loss_precision=0.0,loss_recall=0.0,acos_loss_recall=0.0,
          loss_fn=0.0)
        start_of_cell=TRUE
      }else{
        df=rbind(df,dv)
        
        start_of_cell=FALSE
      }
    }else if(grepl(".*.mat",line)){
      dv$img_name=line
    }else if(grepl("report.*",line)){
      current_variable="report"
    }else if(grepl("precision.*binary.*",line)){
      # table=read.table(textConnection(object = line),header = FALSE,sep=":",dec='.')
      # precision_binary=table[1,2]
      strs=unlist(strsplit(line,":"))
      dv$precision_binary=as.numeric(strs[2])
    }else if(grepl("recall.*binary.*",line)){
      strs=unlist(strsplit(line,":"))
      dv$recall_binary=as.numeric(strs[2])
    }else if(grepl("precision.*micro*",line)){
      strs=unlist(strsplit(line,":"))
      dv$precision_micro=as.numeric(strs[2])
    }else if(grepl("recall.*micro*",line)){
      strs=unlist(strsplit(line,":"))
      dv$recall_micro=as.numeric(strs[2])
    }else if(grepl("precision.*macro*",line)){
      strs=unlist(strsplit(line,":"))
      dv$precision_macro=as.numeric(strs[2])
    }else if(grepl("recall.*macro*",line)){
      strs=unlist(strsplit(line,":"))
      dv$recall_macro=as.numeric(strs[2])
    }else if(grepl("precision.*weighted*",line)){
      strs=unlist(strsplit(line,":"))
      dv$precision_weighted=as.numeric(strs[2])
    }else if(grepl("recall.*weighted*",line)){
      strs=unlist(strsplit(line,":"))
      dv$recall_weighted=as.numeric(strs[2])
    }else if(grepl("AUC.*",line)){
      strs=unlist(strsplit(line,":"))
      dv$AUC=as.numeric(strs[2])
    }else if(grepl("MAE.*",line)){
      strs=unlist(strsplit(line,":"))
      dv$MAE=as.numeric(strs[2])
    }else if(grepl("CC.*",line)){
      strs=unlist(strsplit(line,":"))
      dv$CC=as.numeric(strs[2])
    }else if(grepl("NSS.*",line)){
      strs=unlist(strsplit(line,":"))
      dv$NSS=as.numeric(strs[2])
    }else if(grepl(".*loss_crossentropy",line)){
      strs=unlist(strsplit(line,"[(]"))     #标点符号作为分隔符这样用，详见:help("strsplit")
      strs=unlist(strsplit(strs[2],","))
      dv$loss_crossentropy=as.numeric(strs[1])
    }else if(grepl("branch1:loss_precision",line)){
      strs=unlist(strsplit(line,"[(]"))     #标点符号作为分隔符这样用，详见:help("strsplit")
      strs=unlist(strsplit(strs[2],"[)]"))
      dv$loss_precision=as.numeric(strs[1])
    }else if(grepl("branch1:acos_loss_precision",line)){
      strs=unlist(strsplit(line,"[(]"))     #标点符号作为分隔符这样用，详见:help("strsplit")
      strs=unlist(strsplit(strs[2],"[)]"))
      dv$acos_loss_precision=as.numeric(strs[1])
    }else if(grepl("branch1:loss_recall",line)){
      strs=unlist(strsplit(line,"[(]"))     #标点符号作为分隔符这样用，详见:help("strsplit")
      strs=unlist(strsplit(strs[2],"[)]"))
      dv$loss_recall=as.numeric(strs[1])
    }else if(grepl("branch1:acos_loss_recall",line)){
      strs=unlist(strsplit(line,"[(]"))     #标点符号作为分隔符这样用，详见:help("strsplit")
      strs=unlist(strsplit(strs[2],"[)]"))
      dv$acos_loss_recall=as.numeric(strs[1])
    }else if(grepl("branch1:loss_fn",line)){
      strs=unlist(strsplit(line,"[(]"))     #标点符号作为分隔符这样用，详见:help("strsplit")
      strs=unlist(strsplit(strs[2],","))
      dv$loss_fn=as.numeric(strs[1])
    }else if(grepl("others.*",line)){
      current_variable="others"
    }else if(grepl("[-]{3,}",line)){
    }else if(grepl("",line)){
    }else if(current_variable != "None"){
      if(current_variable == "report"){
        #暂不处理
      }else if(current_variable == "others"){
        #不在这里做处理
      }
    }
  }
  return(df)
}

#搜索路径，返回想要的路径列表
#location:搜索开始的位置，相对路径或绝对路劲
#pattern:匹配模式(文件夹名称列表，列表中所有的名称必须都出现在路径中才会匹配,
#将想要匹配到的文件或文件夹放第一个，放在后面会搜不到)
search_dir=function(location,patterns = NULL){
  
  if(is.null(patterns)){
    return(NULL)
  }
  
  paths_str=dir(path=location,pattern=patterns[1],recursive = TRUE,include.dirs = TRUE)
  slash_num=str_count(paths_str,"/")  #统计列表中每个元素的斜杠数量
  paths=strsplit(paths_str,split = "/")
  max_len=max(slash_num)+1        #获取组大目录层级
  df=data.frame()
  
  cnt=1
  for(path in paths){
    #求最大向量长度
    vec=rep("|",max_len)
    point=0
    for(i in path){
      point=point+1
      vec[point]=i
    }
    
    # append(vec,paths_str[cnt])                #这样行不通，不知道为什么
    vec=append(vec,paths_str[cnt])
    cnt=cnt+1
    df=rbind(df,vec)
  }
  # # 与运算有用，但是str_detect会经行正则化匹配，不能完全精准匹配
  # sub_df=df%>%filter_all(any_vars(str_detect(.,pattern=patterns[2])&str_detect(.,pattern="epoch1")))
  # #与运算出现bug，但是能够精准匹配
  # sub_df=df%>%filter_all(any_vars((.==patterns[2])&(.==patterns[3])))
  # #或运算有用，能够精准匹配
  # sub_df=df%>%filter_all(any_vars((.==patterns[2])|(.==patterns[3])))
  len=length(patterns)
  if(len>1){
    for (i in 1:(len-1)+1) {
      df=df%>%filter_all(any_vars((.==patterns[i])))
    }
  }
  
  paths=rep(location,nrow(df))
  dirs=paste(paths,df[,ncol(df)],sep = "/")
  return(dirs)
  
}

#搜索路径，返回想要的路径列表
#location:搜索开始的位置，相对路径或绝对路劲
#file:想要搜索的文件夹或文件
#pattern:匹配模式(文件夹名称列表，只要列表中的的一个名称出现在路径中才会匹配，
#不要将想搜索的文件夹或文件名称放入此参数)
search_dir_or_match=function(location,file,patterns){
  
  paths_str=dir(path=location,pattern=file,recursive = TRUE,include.dirs = TRUE)
  slash_num=str_count(paths_str,"/")  #统计列表中每个元素的斜杠数量
  paths=strsplit(paths_str,split = "/")
  max_len=max(slash_num)+1        #获取组大目录层级
  df=data.frame()
  
  cnt=1
  for(path in paths){
    #求最大向量长度
    vec=rep("|",max_len)
    point=0
    for(i in path){
      point=point+1
      vec[point]=i
    }
    # append(vec,paths_str[cnt])                #这样行不通，不知道为什么
    vec=append(vec,paths_str[cnt])
    cnt=cnt+1
    df=rbind(df,vec)
  }
  # # 与运算有用，但是str_detect会经行正则化匹配，不能完全精准匹配
  # df=df%>%filter_all(any_vars(str_detect(.,pattern=patterns[2])&str_detect(.,pattern="epoch1")))
  # #与运算出现bug，但是能够精准匹配
  # df=df%>%filter_all(any_vars((.==patterns[2])&(.==patterns[3])))
  #或运算有用，能够精准匹配
  # df=df%>%filter_all(any_vars((.==patterns[2])|(.==patterns[3])))
  len=length(patterns)
  if(len>1){
    cmd="df=df%>%filter_all(any_vars((.=='"
    cmd_middle=paste(patterns[1:len],collapse = "')|(.=='")
    cmd=paste(cmd,cmd_middle,"')))",sep = "")
    cmd=parse(text=cmd)
    eval(cmd)
  }else if(len==1){
    df=df%>%filter_all(any_vars((.==patterns[1])))
  }
  paths=rep(location,nrow(df))
  dirs=paste(paths,df[,ncol(df)],sep = "/")
  return(dirs)
}


#搜索路径，返回想要的路径数据框
#location:搜索开始的位置，相对路径或绝对路劲
#pattern:匹配模式(文件夹名称列表，列表中所有的名称必须都出现在路径中才会匹配,
#将想要匹配到的文件或文件夹放第一个，放在后面会搜不到)
search_dir_return_df=function(location,patterns,subdir=TRUE){
  
  paths_str=dir(path=location,pattern=patterns[1],recursive = subdir,include.dirs = TRUE)
  slash_num=str_count(paths_str,"/")  #统计列表中每个元素的斜杠数量
  paths=strsplit(paths_str,split = "/")
  max_len=max(slash_num)+1        #获取组大目录层级
  df=data.frame()
  
  # ========================================================
  col_names=rep("1",max_len+1)
  for(i in 1:(max_len+1)){
    col_names[i]=paste("col",i,sep = "_")
  }
  # ========================================================
  cnt=1
  for(path in paths){
    #求最大向量长度
    vec=rep("|",max_len)
    point=0
    for(i in path){
      point=point+1
      vec[point]=i
    }
    
    # append(vec,paths_str[cnt])                #这样行不通，不知道为什么
    vec=append(vec,paths_str[cnt])
    cnt=cnt+1
    df=rbind(df,vec)
    # ========================================================
    colnames(df)=col_names
    # ========================================================
  }
  len=length(patterns)
  if(len>1){
    for (i in 1:(len-1)+1) {
      df=df%>%filter_all(any_vars((.==patterns[i])))
    }
  }
  
  paths=rep(location,nrow(df))
  dirs=paste(paths,df[,ncol(df)],sep = "/")
  df[,ncol(df)]=dirs
  return(df)
}

#搜索路径，返回想要的路径数据框
#location:搜索开始的位置，相对路径或绝对路劲
#file:想要搜索的文件夹或文件
#pattern:匹配模式(文件夹名称列表，只要列表中的的一个名称出现在路径中才会匹配，
#不要将想搜索的文件夹或文件名称放入此参数)
search_dir_or_match_return_df=function(location,file,patterns){
  
  paths_str=dir(path=location,pattern=file,recursive = TRUE,include.dirs = TRUE)
  slash_num=str_count(paths_str,"/")  #统计列表中每个元素的斜杠数量
  paths=strsplit(paths_str,split = "/")
  max_len=max(slash_num)+1        #获取组大目录层级
  df=data.frame()
  # ========================================================
  col_names=rep("1",max_len+1)
  for(i in 1:(max_len+1)){
    col_names[i]=paste("col",i,sep = "_")
  }
  # ========================================================
  cnt=1
  for(path in paths){
    #求最大向量长度
    vec=rep("|",max_len)
    point=0
    for(i in path){
      point=point+1
      vec[point]=i
    }
    # append(vec,paths_str[cnt])                #这样行不通，不知道为什么
    vec=append(vec,paths_str[cnt])
    cnt=cnt+1
    df=rbind(df,vec)
    # ========================================================
    colnames(df)=col_names
    # ========================================================
  }
  len=length(patterns)
  if(len>1){
    cmd="df=df%>%filter_all(any_vars((.=='"
    cmd_middle=paste(patterns[1:len],collapse = "')|(.=='")
    cmd=paste(cmd,cmd_middle,"')))",sep = "")
    cmd=parse(text=cmd)
    eval(cmd)
  }else if(len==1){
    df=df%>%filter_all(any_vars((.==patterns[1])))
  }
  paths=rep(location,nrow(df))
  dirs=paste(paths,df[,ncol(df)],sep = "/")
  df[,ncol(df)]=dirs
  return(df)
}

# file="D:/workspace/python_workspace/MBCNN/MBCNN/net/End_to_End_L5_BN1/End_to_End_L5_BN1_No.1/result/epoch1/evaluation.txt"
# print(search_dir_or_match_return_df("./net","evaluation.txt",c("End_to_End_L5_BN1_No.1","End_to_End_L5_BN1_No.2")))

# file="D:/workspace/python_workspace/MBCNN/MBCNN/net/End_to_End_L5_BN1/End_to_End_L5_BN1_No.1/result/epoch1/evaluation.txt"
# print(search_dir_return_df("./net",c("evaluation.txt","End_to_End_L5_BN1_No.1")))

# file="D:/workspace/python_workspace/MBCNN/MBCNN/net/End_to_End_L5_BN1/End_to_End_L5_BN1_No.1/result/epoch1/evaluation.txt"
# print(search_dir_or_match("./net","evaluation.txt",c("End_to_End_L5_BN1_No.1","End_to_End_L5_BN1_No.2")))

# file="D:/workspace/python_workspace/MBCNN/MBCNN/net/End_to_End_L5_BN1/End_to_End_L5_BN1_No.1/result/epoch1/evaluation.txt"
# print(search_dir("./net",c("evaluation.txt","End_to_End_L5_BN1_No.1","epoch1","result")))

# file="D:/workspace/python_workspace/MBCNN/MBCNN/net/End_to_End_L5_BN1/End_to_End_L5_BN1_No.1/result/epoch1/evaluation.txt"
# print(read_file_with_numeric(file))

