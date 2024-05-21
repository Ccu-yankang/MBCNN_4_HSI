import random
import FileSys as fs
import Process as process
import net
import Evaluate as eva
import torch

branch_num=1
net_layer=5
net_name="End_to_End_L{}_BN{}".format(net_layer,branch_num)
epoch=50
#initial
#=======================================================================================================================
stream=fs.Stream(net_name=net_name,No=1,epoch=1,file_name="information.txt")
stream.initial()
p=process.process()
# use_cuda=torch.cuda.is_available()
for No in range(20):
    No+=4
    #trainning and test image list
    #=======================================================================================================================
    imagelist=p.get_imageList(stream.scan_imageList())
    stream.set_parameters(net_name=net_name,No=No,epoch=1,file_name="information.txt",first_level_path="net")
    stream.write(imagelist)

    #sample
    #=======================================================================================================================
    strs=""
    channels=p.get_sample(samples=54,branch_num=1)
    for ch in channels:
        strs=strs+str(ch)+"\n"
    stream.set_parameters(net_name=net_name, No=No, epoch=1, file_name="channel.txt", first_level_path="net")
    stream.write(strs)
    model=net.MBCNN(branch_num=1,channels=channels,use_MSE=True)
    model.set_parameters(lr=0.001,momentum=0.90)
    #=======================================================================================================================

    for i in range(epoch):
        i+=1
        random.shuffle(p.image_train_list)
        for index,image_name in enumerate(p.image_train_list,1):
            #hyperspectral image
            #=======================================================================================================================
            stream.set_parameters(net_name=net_name,No=No,epoch=i,file_name=image_name,first_level_path="dataset")
            image=stream.read()
            image_data=p.img_process(image)
            #ground truth
            #=======================================================================================================================
            jpg_name=image_name.replace(".mat",".jpg")
            stream.set_parameters(net_name=net_name,No=No,epoch=i,file_name=jpg_name,first_level_path="dataset")
            ground_truth=stream.read()
            ground_truth=p.ground_truth_process(ground_truth)
            #trainning MBCNN
            #=======================================================================================================================
            # if use_cuda:
            #     image_data=image_data.cuda()
            #     ground_truth=ground_truth.cuda()
            saliency=model.run(data=image_data,ground_truth=ground_truth)
            img = p.saliency_process(saliency)
            #evaluate
            #=======================================================================================================================
            measure = eva.Evaluate()
            # saliency,ground_truth=p.eva_process(saliency,ground_truth)
            # measure.evaluate(saliency,ground_truth,["class0","class1"])
            measure.add_others(model.loss_info)
            #save model and result
            #=======================================================================================================================
            measure.save_format_when_trainning(image_name)
            stream.set_parameters(net_name=net_name, No=No, epoch=i, file_name="evaluation.txt", first_level_path="net")
            stream.write(measure.strs)
            #路径不存在时，cv2.imwrite()不会报错
            stream.set_parameters(net_name=net_name, No=No, epoch=i, file_name=jpg_name, first_level_path="net")
            stream.write(img)
        for b in range(branch_num):
            model_name=net_name+f"_No.{stream.No}_B{b+1}_epoch{i}.pkl"
            stream.set_parameters(net_name=net_name, No=No, epoch=i, file_name=model_name,
                                  first_level_path="net")
            stream.write(model.model[b])

