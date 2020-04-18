from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image,ImageDraw
import torchvision.transforms as transforms
import torch
import numpy as np
import copy
import cv2
import random
import math
class MyDataset():
    def __init__(self,opts,phase):
        self.opts=opts
        self.phase = phase

        self.dataLocation="dataset"
        # self.transforms=self.getTransforms()
        self.trainset =self.load_image()
        # self.calculateMeanAndStd()
        self.newMethod=True
        print("new Mthod: "+str(self.newMethod))




    #load image dataset
    def load_image(self):
        print(self.phase)
        fileName=os.path.join(self.dataLocation,self.phase+".xlsx")
        data=pd.read_excel(fileName)
        data=data.to_numpy()
        # print(data[0][0])
        return data
    #split image name and box
    def process_pair(self,data_pair):
        if self.phase=="train":
            fileName=data_pair[0]
            # image=Image.open(os.path.join(self.dataLocation,self.phase,fileName))#open image
            #
            #
            # #image transform
            # image=transforms.Resize(self.opts.load_size)(image)
            #
            # image_copy1 =copy.deepcopy(image) # copy image
            # image_copy1=np.array(image_copy1)
            # image_laplacian = cv2.Laplacian(image_copy1,-1, ksize=3)#laplacian transform
            # image_laplacian=Image.fromarray(image_laplacian)
            #
            #
            #
            # image=transforms.ToTensor()(image)#change to tensor
            # image_laplacian=transforms.ToTensor()(image_laplacian)
            # image_maxPool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(image)#max pool get sharpness

            # image, image_laplacian, image_maxPool=self.processChannelAugumentation(fileName)


            # image = torch.cat((image, image_laplacian, image_maxPool), dim=0)
            # image_deepCopy = copy.deepcopy(image)
            # transforms.Normalize([0.168213, 0.020664, 0.184778],
            #                      [0.242981, 0.076148, 0.257743])(image)

            image = Image.open(os.path.join(self.dataLocation, self.phase, fileName))  # open image
            label = data_pair[1]
            label = label + ""
            label_array = label.split(",")
            for i in range(len(label_array)):
                label_array[i] = float(label_array[i])
            test_copy=copy.deepcopy(label_array)
            # self.saveByChannel(image, "testOriginal"+str(self.count), "result", boxes=label_array)
            image=np.array(image)
            image,label_array=self.preCrop(image,0.9,label_array,5)
            image=Image.fromarray(image)

            image = transforms.Resize([self.opts.crop_size,self.opts.crop_size])(image)

            # transforms.Normalize([42.885723 / 255],
            #                      [62.324052 / 255])(image)





            # self.saveByChannel(image, "testBegin"+str(self.count), "result", boxes=label_array)
            # print(label_array)
            # self.count = 0
            index = random.randint(1, 4)
            if index == 2:
                image=image.transpose(Image.FLIP_LEFT_RIGHT)
                label_array=self.horizontal(label_array)
            elif index == 3:
                image =image.transpose(Image.FLIP_TOP_BOTTOM)
                label_array =self.vertical(label_array)
            elif index == 4:
                image =image.transpose(Image.FLIP_LEFT_RIGHT)
                label_array =self.horizontal(label_array)
                image =image.transpose(Image.FLIP_TOP_BOTTOM)
                label_array =self.vertical(label_array)
            # print("New Label: ",label_array)
            # print(index)
            #
            # self.saveByChannel(image, "test"+str(index)+"-"+str(self.count), "result", boxes=label_array)
            # self.count=self.count+1

            image = transforms.ToTensor()(image)  # change to tensor
            # image transform
            transforms.Normalize([0.168213],
                                 [0.242981])(image)





            label_array_new = []
            self.area=(label_array[2]-label_array[0])*(label_array[3]-label_array[1])
            label_array_new.append([label_array[0], label_array[1], label_array[2], label_array[3]])
            box = torch.as_tensor(label_array_new)
            # self.saveByChannel(image_deepCopy, "initial", "result", boxes=box)
            # self.saveByChannel(image, "test", "result", boxes=box)





            class_label = torch.ones((1),dtype=torch.int64)

        else:
            fileName = data_pair[0]

            image = Image.open(os.path.join(self.dataLocation, self.phase, fileName))  # open image
            label = data_pair[1]
            label = label + ""
            label_array = label.split(",")
            for i in range(len(label_array)):
                label_array[i] = float(label_array[i])
            test_copy = copy.deepcopy(label_array)
            # self.saveByChannel(image, "testOriginal"+str(self.count), "result", boxes=label_array)
            image = np.array(image)
            image, label_array = self.preCrop(image, 0.8, label_array)
            image = Image.fromarray(image)

            image = transforms.Resize([self.opts.crop_size, self.opts.crop_size])(image)

            image = transforms.ToTensor()(image)  # change to tensor
            # image transform
            transforms.Normalize([0.168213],
                                 [0.242981])(image)

            label_array_new = []
            self.area = (label_array[2] - label_array[0]) * (label_array[3] - label_array[1])
            label_array_new.append([label_array[0], label_array[1], label_array[2], label_array[3]])
            box = torch.as_tensor(label_array_new)
            # self.saveByChannel(image_deepCopy, "initial", "result", boxes=box)
            # self.saveByChannel(image, "test", "result", boxes=box)

            class_label = torch.ones((1), dtype=torch.int64)

        return image, box, class_label
    def processChannelAugumentation(self,fileName):
        image = Image.open(os.path.join(self.dataLocation, self.phase, fileName))  # open image

        # image transform
        image = transforms.Resize(self.opts.load_size)(image)

        image_copy1 = copy.deepcopy(image)  # copy image
        image_copy1 = np.array(image_copy1)
        image_laplacian = cv2.Laplacian(image_copy1, -1, ksize=3)  # laplacian transform
        image_laplacian = Image.fromarray(image_laplacian)

        image = transforms.ToTensor()(image)  # change to tensor
        image_laplacian = transforms.ToTensor()(image_laplacian)
        image_maxPool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(image)  # max pool get sharpness

        return image,image_laplacian,image_maxPool

    def __getitem__(self, index):
        data_pair=self.trainset[index]
        image,box,label=self.process_pair(data_pair)
        target={}
        target["boxes"]=box
        target["labels"]=label
        # target["image_id"] = torch.tensor([index])
        # target["area"] = torch.tensor(self.area)
        # target["iscrowd"] = torch.zeros((1))

        return image,target
    def __len__(self):
        return len(self.trainset)

    #transform
    def getTransforms(self):
        transform=[]
        transform.append(transforms.ToTensor())

        transform+=[transforms.Normalize([42.885723/255,42.885723/255,42.885723/255],[62.324052/255,62.324052/255,62.324052/255])]
        # transform += [transforms.Normalize([42.885723 / 255],
        #                                    [62.324052 / 255])]

        transform=transforms.Compose(transform)
        return transform

    #calculate dataset mean and std. Use the value for normalization
    def calculateMeanAndStd(self):
        R_channel=0
        G_channel=0
        B_channel=0
        w=0
        h=0

        for i in range(len(self.trainset)):
            fileName = self.trainset[0][0]
            image = Image.open(os.path.join(self.dataLocation, self.phase, fileName))

            w, h = image.size
            image=np.array(image)
            R_channel=R_channel+np.sum(image[:,:])
        size=len(self.trainset)*w*h
        R_mean=R_channel/size
        R_std=0
        for i in range(len(self.trainset)):
            fileName = self.trainset[0][0]
            image = np.array(Image.open(os.path.join(self.dataLocation, self.phase, fileName)))
            R_std=R_std+np.sum((image[:,:]-R_mean)**2)

        R_std=R_std/size
        print("Mean: [%f]"%(R_mean))
        print("Std: [%f]"%(np.sqrt(R_std)))


    def calculateMeanAndStdThreeChannel(self):
        R_channel=0
        R2_channel=0
        R3_channel=0
        w=0
        h=0

        w2=0
        h2=0

        w3=0
        h3=0
        for i in range(len(self.trainset)):
            fileName = self.trainset[0][0]
            image = Image.open(os.path.join(self.dataLocation, self.phase, fileName))
            # self.drawRectangle(image)
            # image.save("Test.png")
            image = transforms.Resize(self.opts.load_size)(image)

            image_copy1 = copy.deepcopy(image)  # copy image
            image_copy1 = np.array(image_copy1)
            image_laplacian = cv2.Laplacian(image_copy1,-1, ksize=3,)  # laplacian transform
            image_laplacian = Image.fromarray(image_laplacian)

            image = transforms.ToTensor()(image)  # change to tensor
            image_laplacian=transforms.ToTensor()(image_laplacian)
            image_maxPool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(image) # max pool get sharpness



            # if(image.size!=image_laplacian.size or image.size!=image_maxPool.size):
            #     print("Problem!")
            # print(image.size())
            # print(image_laplacian.size())C
            # print(image_maxPool.size())
            # image=transforms.ToPILImage()(image)
            # image_laplacian=transforms.ToPILImage()(image_laplacian)
            # image_maxPool=transforms.ToPILImage()(image_maxPool)
            # image.save("1.png")
            # image_laplacian.save("2.png")
            # image_maxPool.save("3.png")
            # input("C")

            # image=image.numpy()
            # image_laplacian=image_laplacian.numpy()
            # image_maxPool=image_maxPool.numpy()
            # # print(image)
            # # print(image_laplacian)
            # # print(image_maxPool)
            # # input("")
            # R_channel=R_channel+np.sum(image[:,:])
            # R2_channel = R2_channel + np.sum(image_laplacian[:, :])
            # R3_channel = R3_channel + np.sum(image_maxPool[:, :])

            # image = image.numpy()
            # image_laplacian = image_laplacian.numpy()
            # image_maxPool = image_maxPool.numpy()
            R_channel = R_channel + torch.sum(image)
            R2_channel = R2_channel + torch.sum(image_laplacian)
            R3_channel = R3_channel + torch.sum(image_maxPool)

        size=float(len(self.trainset))*float(self.opts.crop_size)*float(self.opts.crop_size)
        print(size)
        print(R2_channel)
        R_mean=R_channel/size
        R2_mean=R2_channel/size
        print(R2_mean)
        R3_mean=R3_channel/size
        R_std=0
        R2_std=0
        R3_std=0
        for i in range(len(self.trainset)):
            fileName = self.trainset[0][0]

            image = Image.open(os.path.join(self.dataLocation, self.phase, fileName))
            # self.drawRectangle(image)
            # image.save("Test.png")
            image = transforms.Resize(self.opts.load_size)(image)

            image_copy1 = copy.deepcopy(image)  # copy image
            image_copy1 = np.array(image_copy1)
            image_laplacian = cv2.Laplacian(image_copy1, -1, ksize=3, )  # laplacian transform
            image_laplacian = Image.fromarray(image_laplacian)

            image = transforms.ToTensor()(image)  # change to tensor
            image_laplacian = transforms.ToTensor()(image_laplacian)
            image_maxPool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(image)  # max pool get sharpness

            R_std=R_std+torch.sum((image-R_mean)**2)
            R2_std=R2_std+torch.sum((image_laplacian-R2_mean)**2)
            R3_std = R3_std + torch.sum((image_maxPool - R3_mean) ** 2)


        R_std=R_std/size
        R2_std = R2_std / size
        R3_std = R3_std / size
        print("Mean: [%f][%f][%f]"%(R_mean,R2_mean,R3_mean))
        print("Std: [%f][%f][%f]"%(np.sqrt(R_std),np.sqrt(R2_std),np.sqrt(R3_std)))

    def drawRectangle(self,image,boxes=None):
        if boxes is None:
            img1=ImageDraw.Draw(image)
            img1.rectangle([(257,200),(291,231)],outline="red")
        else:
            img1 = ImageDraw.Draw(image)
            img1.rectangle([(boxes[0],boxes[1]), (boxes[2], boxes[3])], outline="red",width=3)
        return img1
    def saveByChannel(self,image,name,dic,boxes=None):

        name_tep=str(name)+'.png'
        name_tep=os.path.join(dic,name_tep)
        image_tmp=image
        if boxes is not None:
            self.drawRectangle(image_tmp,boxes)
        image_tmp.save(name_tep)
    def horizontal(self,boxes):
        tepXMin=boxes[0]
        tepXMax=boxes[2]

        middle=self.opts.load_size/2+0.5
        newXMax=middle-tepXMin+middle
        newXMin=middle-tepXMax+middle

        boxes[0]=newXMin
        boxes[2]=newXMax


        return boxes
    def vertical(self,boxes):
        tepYMin=boxes[1]
        tepYMax=boxes[3]

        middle=self.opts.load_size/2+0.5
        newYMax=middle-tepYMin+middle
        newYMin=middle-tepYMax+middle

        boxes[1]=newYMin
        boxes[3]=newYMax
        return boxes

    def preCrop(self,image,coefficient,box,threshold):
        top=0
        size=image.shape[0]
        for i in range(size):
            count=0.0
            for j in range(size):
                if image[i][j]<=threshold:
                    count=count+1

            if count/size<coefficient:
                top=i
                break
        button=size-1
        for i in range(size):
            count=0.0
            for j in range(size):
                if image[size-i-1][j]<=threshold:
                    count=count+1
            if count/size<coefficient:
                button=size-i-1
                break

        left=0
        for i in range(size):
            count=0.0
            for j in range(size):
                if image[j][i]<=threshold:
                    count=count+1
            if count/size<coefficient:
                left=i
                break
        right=0
        for i in range(size):
            count=0.0
            for j in range(size):
                if image[j][size-i-1]<=threshold:
                    count=count+1
            if count/size<coefficient:
                right=size-i-1
                break
        # print(box)
        # print("Top: ",top)
        # print("Button",button)
        # print("Left: ",left)
        # print("Right ",right)

        if(self.phase=="train"):
            box[0]=(min(max(0,box[0]-left),right-left))*self.opts.crop_size/(right-left)
            box[1]=(min(max(0,box[1]-top),button-top))*self.opts.crop_size/(button-top)
            box[2]=(min(max(0,box[2]-left),right-left))*self.opts.crop_size/(right-left)
            box[3]=(min(max(0,box[3]-top),button-top))*self.opts.crop_size/(button-top)
        else:
            box[0] = ( box[0] - left) * self.opts.crop_size / (right - left)
            box[1] = (box[1] - top) * self.opts.crop_size / (button - top)
            box[2] = ( box[2] - left) * self.opts.crop_size / (right - left)
            box[3] = ( box[3] - top) * self.opts.crop_size / (button - top)
        # print(box)

        return image[top:button+1,left:right+1],box





