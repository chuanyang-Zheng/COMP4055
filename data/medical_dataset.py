from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import copy
class MyDataset():
    def __init__(self,opts,phase):
        self.opts=opts
        self.phase = phase

        self.dataLocation="dataset"
        # self.transforms=self.getTransforms()
        self.trainset =self.load_image()
        # self.calculateMeanAndStd()



    #load image dataset
    def load_image(self):
        print(self.phase)
        fileName=os.path.join(self.dataLocation,self.phase+".xlsx")
        data=pd.read_excel(fileName)
        data=data.to_numpy()
        print(data[0][0])
        return data
    #split image name and box
    def process_pair(self,data_pair):
        fileName=data_pair[0]
        image=Image.open(os.path.join(self.dataLocation,self.phase,fileName))


        #image transform
        image=transforms.Resize(self.opts.load_size)(image)
        image=transforms.ToTensor()(image)
        image = torch.cat((image, image, image), dim=0)

        transforms.Normalize([42.885723 / 255, 42.885723 / 255, 42.885723 / 255],
                             [62.324052 / 255, 62.324052 / 255, 62.324052 / 255])(image)

        # transforms.Normalize([42.885723 / 255],
        #                      [62.324052 / 255])(image)

        label=data_pair[1]
        label=label+""
        label_array=label.split(",")
        for  i in range(len(label_array)):
            label_array[i]=float(label_array[i])

        label_array_new = []
        self.area=(label_array[2]-label_array[0])*(label_array[3]-label_array[1])
        label_array_new.append([label_array[0], label_array[1], label_array[2], label_array[3]])
        box = torch.as_tensor(label_array_new)
        coefficient = float(self.opts.crop_size) / float(512)
        box = box * coefficient


        class_label = torch.ones((1),dtype=torch.int64)

        return image, box, class_label

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

