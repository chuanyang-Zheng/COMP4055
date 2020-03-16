from options.train_options import TrainOptions
from data.medical_dataset import *
from torch.utils.data import DataLoader
from models.medical_model import *
from utilS.util import *
import numpy as np
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import  torchvision
from engine import train_one_epoch, evaluate
import utils

if __name__=="__main__":
    count=0
    opts=TrainOptions().parse()
    torch.cuda.set_device(opts.gpu_ids[0])
    device = torch.device("cuda:{}".format(opts.gpu_ids[0])) if len(opts.gpu_ids) else torch.device("cpu")

    dataset=MyDataset(opts,"train")
    data_size=len(dataset)
    dataset=DataLoader(dataset,batch_size=opts.batch_size,shuffle=not opts.serial_batches,num_workers=int(opts.num_threads),collate_fn=utils.collate_fn)

    model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes=2
    in_feature=model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feature, num_classes)
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels= 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)
    model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    # model=Detection(opts)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    scheduler=makeOptimizerAndScheduler(opts,optimizer)

    model=model.to(device)
    model=nn.DataParallel(model,[opts.gpu_ids[0]])


    recorder=Recorder(opts)
    recorder.record("---Load %d images ---"%data_size,False)

    dataVal=MyDataset(opts,"test")
    dataValLoader=DataLoader(dataVal,batch_size=1,shuffle=not opts.serial_batches,num_workers=int(opts.num_threads))

    for epoch in range(opts.epoch_count, opts.n_epochs + opts.n_epochs_decay + 1):
        # for i, data in enumerate(dataset):
        #     optimizer.zero_grad()
        #     images=data[0].to(device)
        #     targets=data[1]
        #     # print(targets.size)

        train_one_epoch(model, optimizer, dataset, device, epoch, print_freq=10)
        loss=0
        # print(output)

        # for i in output:
        #     loss+=output[i]
        # loss.backward()
        # optimizer.step()


        scheduler.step()

        # evaluate(model,dataValLoader, device=device)
        recorder.record("Finish [%d/%d]"%(epoch,opts.n_epochs+opts.n_epochs_decay))
        recorder.record("IOU Accuracy is: {}".format(calculateDatasetIOU(dataValLoader,model,device,len(dataVal))))
        model.train()
        if epoch%opts.save_epoch_freq==0:
            saveNetwork(model,opts,epoch)





    # recorder=Recorder(opts)
    #
    #
    # dataVal=MyDataset(opts,"test")
    # dataValLoader=DataLoader(dataVal,batch_size=1,shuffle=not opts.serial_batches,num_workers=int(opts.num_threads))
    #
    #
    # recorder.record("---Load %d images ---"%data_size,False)

    #calculate mean and std
    # mean = 0.
    # std = 0.
    # nb_samples = 0.
    #
    # iter_count=0
    # pred=0
    # target=0
    # #train
    # for epoch in range(opts.epoch_count,opts.n_epochs+opts.n_epochs_decay+1):
    #     for i,data in enumerate(dataset):
    #         optimizer.zero_grad()
    #         pred=model(data[0])
    #         target=data[1]["boxes"].to(device)
    #         loss = smooth_l1_loss(pred, target)
    #         loss.backward()
    #         optimizer.step()
    #         iter_count+=data[0].size(0)
    #         if (iter_count%opts.display_freq)==0:
    #             recorder.record("Epoch/Iter [%d][%d] Loss: %f"%(epoch,iter_count%data_size,loss))
    #     recorder.record("Finish [%d/%d]"%(epoch,opts.n_epochs+opts.n_epochs_decay))
    #     recorder.record("IOU Accuracy is: {}".format(calculateDatasetIOU(dataValLoader,model,device,len(dataVal))))
    #     model.train()
    #     model.saveNetwork("latest")
    #     if(epoch%opts.save_epoch_freq)==0:
    #         model.saveNetwork(epoch)
    #     scheduler.step()
    #
    # recorder.record("--- Finish ---")


