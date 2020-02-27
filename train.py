from options.train_options import TrainOptions
from data.medical_dataset import *
from torch.utils.data import DataLoader
if __name__=="__main__":
    opts=TrainOptions().parse()
    dataset=MyDataset(opts)
    dataset=DataLoader(dataset,batch_size=opts.batch_size,shuffle=not opts.serial_batches,num_workers=int(opts.num_threads))
    print("---Load %d images ---"%len(dataset))

    #calculate mean and std
    mean = 0.
    std = 0.
    nb_samples = 0.

    #train
    for i,data in enumerate(dataset):
        mean+=1
