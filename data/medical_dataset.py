from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
class MyDataset():
    def __init__(self,opts):
        self.opts=opts
        self.dataLocation="dataset"

        self.trainset =self.load_image()

    def load_image(self):
        fileName=os.path.join(self.dataLocation,self.opts.phase+".xlsx")
        data=pd.read_excel(fileName)
        data=data[1:]
    def process_pair(self,data_pair):
        fileName=data_pair[0]
        image=Image.open(os.path.join(self.dataLocation,self.opts.phase,fileName))

        label=data_pair[1]

        return image,label
    def __getitem__(self, index):
        data_pair=self.trainset[index]
        image,label=self.process_pair(data_pair)

        return image,label
    def __len__(self):
        return self.trainset.size()[0]
