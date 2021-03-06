import torch
import torch.nn as nn
import math
from torch.nn import init
from torch.optim import lr_scheduler
import os
import numpy as np
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import  torchvision
from models.fpn import *
def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

#From https://github.com/YunzhuLi/VisGel/blob/master/models.py
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample=downsample
        if downsample or stride!=1:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes,planes,stride=stride,kernel_size=1),nn.BatchNorm2d(planes))
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample :
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):


    def __init__(self, inplanes, planes, stride=1, downsample=None,expansion=4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)



        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)



        out = self.conv3(out)
        out = self.bn3(out)



        if self.downsample is not None:
            residual = self.downsample(x)



        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    outDefined = 2048

    def __init__(self, block, layers,expansion=4):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)


        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#128
        self.layer0=self._make_layer(block,64,2,stride=2,expansion=1)

        self.layer1 = self._make_layer(block, 64, layers[0],expansion=expansion)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,expansion=expansion)#64
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,expansion=expansion)#32
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,expansion=expansion)#16
        self.layer5 = self._make_layer(block, 512, layers[4], stride=2,expansion=expansion)#8
        # self.layer6=self._make_layer(block,512,layers[5],stride=2)

        self.sigmoid=nn.Sigmoid()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_final=nn.Conv2d(512*expansion,256,kernel_size=1)
        # self.button = nn.Linear(self.outDefined + 2, 2)

        self.Linear=nn.Linear(256,64)
        self.Linear2 = nn.Linear(64, 16)
        self.Linear3=nn.Linear(16,4)


    def _make_layer(self, block, planes, blocks, stride=1,expansion=4):
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,expansion=expansion))
        self.inplanes = planes * expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,expansion=expansion))

        return nn.Sequential(*layers)

    def forward(self,x):
        result=self.conv1(x)
        result=self.bn1(result)
        result=self.relu(result)



        # result=self.maxpool(result)
        result=self.layer0(result)


        result=self.layer1(result)
        result=self.layer2(result)
        result=self.layer3(result)
        result=self.layer4(result)
        result=self.layer5(result)
        # result=self.layer6(result)

        self.feature1=self.pool(result)

        # self.feature2 = self.feature1.view(self.feature1.shape[0], -1)
        # self.feature3 = self.relu(self.bn(self.feature2))
        box = self.conv_final(self.feature1)

        box=self.Linear3(self.Linear2(self.Linear(box.view(box.shape[0],-1))))
        # button = self.sigmoid(self.button(torch.cat((self.feature3, top), dim=1)))
        #
        # box=torch.cat((top,button),dim=1)


        return box


#/From https://github.com/YunzhuLi/VisGel/blob/master/models.py





def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    print("Create %s Learning Scheduler"%opt.lr_policy)
    return scheduler

def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()

class Detection(nn.Module):


    def __init__(self,opts):
        super(Detection, self).__init__()

        self.opts=opts
        self.encoder=ResNet(Bottleneck,layers=[3, 4,6,4,3])
        self.encoder=init_net(self.encoder,gpu_ids=opts.gpu_ids)






    def forward(self,input):
        return self.encoder(input)

    def getLoss(self):
        return self.loss
    def saveNetwork(self,epoch):
        save_file="%s_Detection.pth"%(epoch)
        save_file=os.path.join(self.opts.checkpoints_dir,self.opts.name,save_file)
        if len(self.opts.gpu_ids)>0:
            torch.save(self.encoder.module.cpu().state_dict(),save_file)
            self.encoder.cuda(self.opts.gpu_ids[0])
        else:
            torch.save(self.encoder.cpu().state_dict(),save_file)


def makeOptimizerAndScheduler(opts,optimizer):
    scheduler = get_scheduler(optimizer, opts)
    return scheduler

def saveNetwork(model,opts,epoch):
    save_file="%s_Detection.pth"%(epoch)
    save_file=os.path.join(opts.checkpoints_dir,opts.name,save_file)
    if len(opts.gpu_ids)>0:
        torch.save(model.module.cpu().state_dict(),save_file)
        model.cuda(opts.gpu_ids[0])
    else:
        torch.save(model.cpu().state_dict(),save_file)

class AdditionalModel(nn.Module):
    def __init__(self,opts):
        super(AdditionalModel, self).__init__()

        channel=3

        self.conv1 = nn.Conv2d(1,channel , kernel_size=7, stride=1, padding=3)
        self.conv1_bn=nn.BatchNorm2d(channel)

        self.conv2_1_3 = BasicBlock(channel,channel,stride=2,downsample=True)
        self.conv1_1_2=BasicBlock(channel,channel,stride=2,downsample=True)

        self.conv1_2_1=BasicBlock(channel,channel,1)
        self.conv2_2_2=BasicBlock(channel,channel,1)
        self.conv3_2_3=BasicBlock(channel,channel,1)

        self.conv3_3_1=nn.ConvTranspose2d(channel,channel,stride=4,kernel_size=4)
        self.con3_3_1_bn=nn.BatchNorm2d(channel)
        self.conv3_3_2=nn.ConvTranspose2d(channel,channel,stride=2,kernel_size=2)
        self.con3_3_2_bn = nn.BatchNorm2d(channel)
        self.conv3_3_3=BasicBlock(3*channel,channel,1,True)

        self.conv2_3_1=nn.ConvTranspose2d(channel,channel,stride=2,kernel_size=2)
        self.conv2_3_1_bn=nn.BatchNorm2d(channel)
        self.conv2_3_3=BasicBlock(channel,channel,2,downsample=True)
        self.conv2_3_2=BasicBlock(3*channel,channel,1,True)

        self.conv1_3_3=BasicBlock(channel,channel,4,True)
        self.conv1_3_2=BasicBlock(channel,channel,2,True)
        self.conv1_3_1=BasicBlock(3*channel,channel,1,True)

        self.end1=BasicBlock(channel,1,1,True)
        self.end2=BasicBlock(channel,1,2,True)
        self.end3=BasicBlock(channel,1,4,True)

        # self.combine=BasicBlock(3*channel,3,1,True)

        self.leaklyU = nn.LeakyReLU(0.2)
    def forward(self, image):
        # out_channel1 = self.conv1(image)
        # out_channel2 = self.conv2(torch.cat((image, out_channel1), dim=1))
        #
        # combine = torch.cat((image, out_channel1, out_channel2), dim=1)
        # preprocess = self.bn(combine)
        # preprocess = self.leaklyU(preprocess)
        image3_1=self.leaklyU(self.conv1_bn(self.conv1(image)))

        image2_1=self.conv2_1_3(image3_1)
        image1_1=self.conv1_1_2(image2_1)

        image3_2=self.conv3_2_3(image3_1)
        image2_2=self.conv2_2_2(image2_1)
        image1_2=self.conv1_2_1(image1_1)

        image3_3_1=self.leaklyU(self.con3_3_1_bn(self.conv3_3_1(image1_2)))
        image3_3_2=self.leaklyU(self.con3_3_2_bn(self.conv3_3_2(image2_2)))
        image3_3=self.conv3_3_3(torch.cat((image3_3_1,image3_3_2,image3_2),dim=1))

        image2_3_1=self.leaklyU(self.conv2_3_1_bn(self.conv2_3_1(image1_2)))
        image2_3_3=self.conv2_3_3(image3_3)
        image2_3=self.conv2_3_2(torch.cat((image2_3_1,image2_2,image2_3_3),dim=1))

        image1_3=self.conv1_3_1(torch.cat((image1_2,self.conv1_3_2(image2_3),self.conv1_3_3(image3_3)),dim=1))

        # preprocess=self.combine(torch.cat((self.end1(image1_3),self.end2(image2_3),self.end3(image3_3)),dim=1))
        end1=self.end1(image1_3)
        end2=self.end2(image2_3)
        end3=self.end3(image3_3)
        preprocess=torch.cat((end1,end2,end3),dim=1)
        # print(preprocess.size())
        # preprocess=self.combine(preprocess)



        return preprocess




class Model(nn.Module):
    def __init__(self,opts):
        super(Model, self).__init__()
        self.opts=opts
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = 2
        in_feature = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feature, num_classes)
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                        output_size=7,
                                                        sampling_ratio=2)
        self.model = FasterRCNN(backbone,
                           num_classes=2,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
        self.preprocess=FPN101()
        init_net(self.preprocess)



    def forward(self, image,target=None):

        #If train, use the following. As we use engine class function when training and the function change image to list, we use torch.stack
        if target is not None:
            take_image=torch.stack(image,dim=0)
            # preprocess=torch.cat((take_image,take_image,take_image),dim=1)
            preprocess=self.preprocess(take_image)
            return self.model(preprocess, target)

        #When Test
        # preprocess = torch.cat((image,image,image), dim=1)
        preprocess = self.preprocess(image)
        return self.model(preprocess)




