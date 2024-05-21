#R语言 文件及路径操作
#http://t.zoukankan.com/shanger-p-12175623.html
#location:包含所有epoch的文件夹名称

library(tidyverse)
metrics=c("AUC","MAE","CC","NSS","precision_binary","recall_binary",
          "precision_macro","recall_macro","precision_micro","recall_micro",
          "precision_weighted","recall_weighted","loss_crossentropy",
          "loss_precision","acos_loss_precision","loss_recall",
          "acos_loss_recall","loss_fn")
# metric=c(metrics[2],metrics[11:12],metrics[13:14],metrics[16])
metric=c(metrics[5:6],metrics[11:12])

#读取工程路径下所有的evaluate.txt数据并返回数据并作图
analyse_net_demo=function(){
  source("read_evaluateFile.R")
  paths=dir(pattern="End_to_End_L5_BN1_No.[0-9]{1,}",recursive = TRUE,include.dirs = TRUE)
  for(path in paths){
    if(!grepl("End_to_End_L[1-9]_BN[1-9]_No.[0-9]{1,}_B[1-10]_epoch[0-9]{1,}",path)){
      eva=data.frame()
      netname=str_extract(path,"(?<=BN[1-9]{1,3}\\/).+")
      nettype=str_extract(path,"(?<=net\\/).+(?=\\/End)")
      cpath=paste(".",path,sep = "/")
      files=list.files(path=cpath,pattern = "evaluation.txt",recursive = TRUE)
      result_path=paste(cpath,"result",sep="/")
      evaluate_path=paste(result_path,"evaluate",sep = "/")
      for(file in files){
        #从文件路径中提取epoch值
        epoch=str_extract(file,"(?<=result\\/).+(?=\\/evaluation)")
        epoch=str_extract(epoch,"[0-9]{1,3}")
        epoch=as.numeric(epoch,digits=3)
        #生成evaluation.txt文件路径
        file_dir=paste(cpath,file,sep = "/")
        
        #读取evaluation.txt中的数据并生成数据框(data frame)
        df=read_file_with_numeric(file_dir)
        row_num=nrow(df)
        #在数据框中添加net_name和net_epoch
        vect1=rep(netname,row_num)
        vect2=rep(epoch,row_num)
        df$net_name=vect1
        df$net_epoch=vect2
        eva=rbind(eva,df)
      }
      
      image_names=factor(eva$img_name)
      image_names=levels(image_names)
      # as.data.frame(lapply(eva,as.numeric))
      #每张图片单独绘图
      for(image_name in image_names){
        sub_eva=filter(eva,img_name==image_name)
        graphic=ggplot(sub_eva,aes(net_epoch,AUC))+
          geom_line(color="blue",linewidth=1)+      #折线图
          geom_point(color="blue",size=3)+          #点图
          geom_smooth(color="red")+                 #趋势线
          labs(x="epoch",y="AUC",title = image_name)
        print(graphic)
        #保存,plot：要保存的图片
        #保存图像
        if(!dir.exists(result_path)){
          dir.create(result_path)
        }
        if(!dir.exists(evaluate_path)){
          dir.create(evaluate_path)
        }
        save_name=str_extract(image_name,".*(?=\\.mat)")
        print(save_name)
        save_name=paste("AUC",save_name,sep = "_")
        save_name=paste(save_name,"png",sep = ".")
        print(save_name)
        save_path=paste(evaluate_path,save_name,sep = "/")
        print(save_path)
        ggsave(filename = save_path,device = "png",plot=graphic,dpi = 300)
      }
      # print(eva)
      # eva2=eva
      # eva2$img_name=as.factor(eva2$img_name)
      # graphic=ggplot(eva,aes(net_epoch,AUC,fill=img_name))+
      #   geom_point(size=3,shape=21)
      # print(graphic)
      
    }
    # print(path)
    # break
  }
}

#从路径中获取net_name
net_name_from_path=function(path){
  net_name=strsplit(path,split = "/")
  net_name=unlist(net_name)
  id=which(net_name=="result")
  if(id<2){
    net_name="None"
  }else{
    net_name=net_name[id-1]
  }
  return(net_name)
}
#从路径中获取epoch
epoch_from_path=function(path){
  epoch=str_match(path,"epoch[0-9]{1,}")
  epoch=as.numeric(str_extract(epoch,"[0-9]{1,}"))
  return(epoch)
}

#从路径获取网络根目录
net_root_from_path=function(path){
  net_name=strsplit(path,split = "/")
  net_name=unlist(net_name)
  id=which(net_name=="result")
  if(id<2){
    root=net_name[1]
  }else{
    root=paste(net_name[1:(id-1)],collapse = "/")
  }
  return(root)
}

#画图
plot_graphic=function(df,x_label,y_label,name){
  #=============================================================
  cmd_str="graphic=ggplot(df,aes("
  cmd_str=paste(cmd_str,x_label,",",sep ="" )
  cmd_str=paste(cmd_str,y_label,"))",sep = "")
  # cmd_str=paste(cmd_str,
  # "))+geom_line(color=\"blue\",linewidth=1)+geom_point(color=\"blue\",size=3)+geom_smooth(color=\"red\")+",
  # sep = "")
  # cmd_str=paste(cmd_str,"labs(x=\"",x_label,sep = "")
  # cmd_str=paste(cmd_str,"\",y=\"",y_label,sep = "")
  # cmd_str=paste(cmd_str,"\",title = \"",name,sep = "")
  # cmd_str=paste(cmd_str,"\")",sep = "")
  # cmd=parse(cmd_str)
  # eval(cmd)
  #=============================================================
  print(cmd_str)
  cmd=parse(text=cmd_str)
  eval(cmd)
  # graphic=ggplot(df,aes(net_epoch,AUC))
  graphic=graphic+
    geom_line(color="blue",linewidth=1)+      #折线图
    geom_point(color="blue",size=3)+          #点图
    geom_smooth(color="red")+                 #趋势线
    labs(x=x_label,y=y_label,title = name)
  print(graphic)
  #=============================================================
  return(graphic)
}

#画图
plot_percentage_graphic_multi_varible=function(df,x_label,y_labels,name){
  vec=c(x_label,y_labels)
  df=df[,vec]
  df=df%>%
    pivot_longer(cols = -net_epoch,names_to = "metrics",values_to = "values")
  df=df%>%
    group_by(metrics)%>%
    mutate(percentage=(values/max(abs(values)))*100,max_value=max(abs(values)))
    # mutate(percentage=round(values/max(abs(values)*100,2)))
  graphic=ggplot(df,aes(x=net_epoch,y=percentage,group=metrics,colour=metrics))
  graphic=graphic+
    geom_line(linewidth=1,alpha=0.7)+      #折线图
    geom_point(size=3,alpha=0.7)+          #点图
    # scale_y_continuous(breaks = c(0,1,2,3,4,5,6,7,8,9,10),
    #                    labels = c("0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"))+
    # coord_cartesian(ylim = c(0,10))+
    labs(x=x_label,y=y_labels,title = name)
  print(graphic)
  #=============================================================
  return(graphic)
}

#画图
plot_graphic_multi_varible=function(df,x_label,y_labels,name){
  vec=c(x_label,y_labels)
  df=df[,vec]
  df=df%>%
    pivot_longer(cols = -net_epoch,names_to = "metrics",values_to = "values")
  graphic=ggplot(df,aes(x=net_epoch,y=values,group=metrics,colour=metrics))
  graphic=graphic+
    geom_line(linewidth=1,alpha=0.7)+      #折线图
    geom_point(size=3,alpha=0.7)+          #点图

    labs(x=x_label,y=y_labels,title = name)
  print(graphic)
  #=============================================================
  return(graphic)
}

get_DFfromFile=function(file,net_name,epoch){
  #读取evaluation.txt中的数据并生成数据框(data frame)
  df=read_file_with_numeric(file)
  row_num=nrow(df)
  #在数据框中添加net_name和epoch
  vect1=rep(net_name,row_num)
  vect2=rep(epoch,row_num)
  df$net_name=vect1
  df$net_epoch=vect2
  return(df)
}


mesure_x_y=function(df,x,y,save_path){
  image_names=factor(df$img_name)
  image_names=levels(image_names)
  for(image_name in image_names){
    sub_df=filter(df,img_name==image_name)
    graphic=plot_graphic(sub_df,x,y,image_name)
    #保存图片
    if(!((is.na(save_path))|(is.null(save_path)))){
      if(!dir.exists(save_path)){
        dir.create(save_path)
      }
      
      save_name=str_extract(image_name,".*(?=\\.mat)")
      save_name=paste(save_name,"png",sep = ".")
      save_name=paste(y,save_name,sep = "_")
      save_file=paste(save_path,save_name,sep = "/")
      ggsave(filename = save_file,device = "png",plot=graphic,dpi = 300)
    }
  }
}

#多个指标
mesure_x_ys=function(df,x,ys,save_path){
  image_names=factor(df$img_name)
  image_names=levels(image_names)
  for(image_name in image_names){
    sub_df=filter(df,img_name==image_name)
    graphic=plot_graphic_multi_varible(sub_df,x,ys,image_name)
    #保存图片
    if(!((is.na(save_path))|(is.null(save_path)))){
      if(!dir.exists(save_path)){
        dir.create(save_path)
      }
      
      metrics_name=paste(ys,collapse="_")
      save_name=str_extract(image_name,".*(?=\\.mat)")
      save_name=paste(save_name,"png",sep = ".")
      save_name=paste(metrics_name,save_name,sep = "_")
      #百分比图
      #=============================================================
      # save_name=paste("persentage",save_name,sep = "_")
      #=============================================================
      save_file=paste(save_path,save_name,sep = "/")
      ggsave(filename = save_file,device = "png",plot=graphic,dpi = 300)
    }
  }
}

analyse_net=function(location){
  source("read_evaluateFile.R")
  
  files=search_dir(location,"result")
  net_roots=character()
  for(path in files){
    net_root=net_root_from_path(path)
    net_roots=append(net_roots,net_root)
  }
  for (path in net_roots) {
    files=search_dir(location = path,"evaluation.txt")
    #处理一个网络的数据
    eva=data.frame()
    for(file in files){
      #获取epoch
      epoch=epoch_from_path(file)
      netname=net_name_from_path(file)
      df=get_DFfromFile(file,netname,epoch)
      eva=rbind(eva,df)
    }
    evaluate_dir=paste(path,"result","evaluate",sep = "/")
    mesure_x_ys(eva,"net_epoch",metric,evaluate_dir)
  }
}

analyse_net("./net/temp")

# analyse_net_demo()
