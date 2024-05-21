import FileSys as fs
import Process as process
import net
import Evaluate as eva

net_name="End_to_End_L{}_BN{}".format(5,1)
epoch=5

stream=fs.Stream(net_name=net_name,No=1,epoch=1,file_name="information.txt")
stream.initial()
p=process.process()
#trainning and test image list
#=======================================================================================================================
stream.set_parameters(net_name="End_to_End_L5_BN1",No=1,epoch=1,file_name="information.txt",first_level_path="net")
p.get_imageList(stream.read())
#sample
#=======================================================================================================================
strs=""
channels=p.get_sample(samples=54,branch_num=1)
for ch in channels:
    strs=strs+str(ch)+"\n"
stream.set_parameters(net_name="End_to_End_L5_BN1", No=1, epoch=1, file_name="channel.txt", first_level_path="net")
stream.write(strs)
#hyperspectral image
#=======================================================================================================================
stream.set_parameters(net_name="End_to_End_L5_BN1",No=1,epoch=1,file_name="0001.mat",first_level_path="dataset")
image=stream.read()
image_data=p.img_process(image)
#ground truth
#=======================================================================================================================
stream.set_parameters(net_name="End_to_End_L5_BN1",No=1,epoch=1,file_name="0001.jpg",first_level_path="dataset")
ground_truth=stream.read()
ground_truth=p.ground_truth_process(ground_truth)
#trainning MBCNN
#=======================================================================================================================
model=net.MBCNN(branch_num=1,channels=channels)
model.set_parameters(lr=0.001,momentum=0.90)
saliency=model.run(data=image_data,ground_truth=ground_truth)
#evaluate
#=======================================================================================================================
measure=eva.Evaluate()
measure.evaluate(saliency,ground_truth,["class0","class1"])
#save model
#=======================================================================================================================
net_name="End_to_End_L5_BN{}_No".format(1,1)
# model_name = net_name + "_B{}_epoch{}.pkl" .format(str(i+1),str(epoch + 1))
