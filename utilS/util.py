"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import time
import numpy as np
def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)
    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array
    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

class Recorder():
    def __init__(self,opts):
        self.filePath=os.path.join(opts.checkpoints_dir,opts.name,"train_record.txt")
        open(self.filePath, "a")
        self.record(("------ {} ------".format(self.getTime())),False)
    def record(self,message,time=True):
        with open(self.filePath, "a") as w:
            if time==True:
                message="%s %s"%(self.getTime(),message)
            w.write(message)
            w.write("\n")
        print(message)

    def getTime(self):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
def calculateIOU(boxA,boxB):
    # boxA[0]=min(boxA[0],boxA[2])
    # boxA[2]=max(boxA[0],boxA[2])
    # boxA[1]=min(boxA[1],boxA[3])
    # boxA[3]=max(boxA[1],boxA[3])
    #
    # boxB[0] = min(boxB[0], boxB[2])
    # boxB[2] = max(boxB[0], boxB[2])
    # boxB[1] = min(boxB[1], boxB[3])
    # boxB[3] = max(boxB[1], boxB[3])

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA ) * max(0, yB - yA )
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] ) * (boxA[3] - boxA[1] )
    boxBArea = (boxB[2] - boxB[0] ) * (boxB[3] - boxB[1] )
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value


    return iou

def calculateDatasetIOU(dataloader,model,device,dataSet_size):
    model.eval()
    count=0
    print("Find IOU")
    print(len(dataloader))
    for i,data in enumerate(dataloader):
        with torch.no_grad():


            target=data[1]["boxes"][0][0].cpu().numpy()
            pred=model(data[0].to(device))
            print(pred)

            # for key, values in pred[0].items():
            #     print(key, values)
            box=pred[0]["boxes"]
            if len(box)<1:
                continue
            score=pred[0]["scores"]
            # print(score)
            index=torch.argmax(score)
            # print(index)
            box=box[index]
            print(index)

            print("Pred",pred)
            print("Target",target)

            print(box.cpu().numpy())

            # result=calculateIOU(torch.squeeze(pred[0]["boxes"],0),torch.squeeze(target[0]["boxes"],0))
            result = calculateIOU(box.cpu().numpy(), target)
            if result>0.5:
                count=count+1
    return count/dataSet_size

def processTarget(targets,images,device):
    target_processed=[]

    for i in range(len(images)):
        d = {}
        d['boxes'] = targets["boxes"].to(device)
        d['labels'] = targets["labels"].to(device)
        target_processed.append(d)


    return target_processed