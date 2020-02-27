from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
class MyDataset():
    def __init__(self,opts):
        self.opts=opts
        self.dataLocation="dataset"
        self.transforms=self.getTransforms()
        self.trainset =self.load_image()
        self.calculateMeanAndStd()

    def load_image(self):
        fileName=os.path.join(self.dataLocation,self.opts.phase+".xlsx")
        data=pd.read_excel(fileName)
        data=data.as_matrix()
        print(data[0][0])
        return data
    def process_pair(self,data_pair):
        fileName=data_pair[0]
        image=Image.open(os.path.join(self.dataLocation,self.opts.phase,fileName))
        image=self.transforms(image)

        label=data_pair[1]
        label=label+""
        label_array=label.split(",")
        for  i in range(len(label_array)):
            label_array[i]=float(label_array[i])

        return image,label_array
    def __getitem__(self, index):
        data_pair=self.trainset[index]
        image,label=self.process_pair(data_pair)

        return image,label,data_pair[0]
    def __len__(self):
        return len(self.trainset)
    def getTransforms(self):
        transform=[]
        transform.append(transforms.ToTensor())


        transform=transforms.Compose(transform)
        return transform
    def calculateMeanAndStd(self):
        R_channel=0
        G_channel=0
        B_channel=0
        w=0
        h=0

        for i in range(len(self.trainset)):
            fileName = self.trainset[0][0]
            image = Image.open(os.path.join(self.dataLocation, self.opts.phase, fileName))

            w, h = image.size
            image=np.array(image)
            R_channel=R_channel+np.sum(image[:,:])




        size=len(self.trainset)*w*h
        R_mean=R_channel/size


        R_std=0


        for i in range(len(self.trainset)):
            fileName = self.trainset[0][0]
            image = np.array(Image.open(os.path.join(self.dataLocation, self.opts.phase, fileName)))
            R_std=R_std+np.sum((image[:,:]-R_mean)**2)

        R_std=R_std/size


        print("Mean: [%f]"%(R_mean))
        print("Std: [%f]"%(np.sqrt(R_std)))

