from options.train_options import TrainOptions
from data.medical_dataset import *
from torch.utils.data import DataLoader
from models.medical_model import *
from utils.utils import *
if __name__=="__main__":
    opts=TrainOptions().parse()
    dataset=MyDataset(opts)
    data_size=len(dataset)
    dataset=DataLoader(dataset,batch_size=opts.batch_size,shuffle=not opts.serial_batches,num_workers=int(opts.num_threads))
    model=Detection(opts)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))
    scheduler=makeOptimizerAndScheduler(opts,optimizer)
    device = torch.device("cuda:{}".format(opts.gpu_ids[0])) if len(opts.gpu_ids) else torch.device("cpu")
    recorder=Recorder(opts)

    recorder.record("---Load %d images ---"%data_size,False)

    #calculate mean and std
    mean = 0.
    std = 0.
    nb_samples = 0.

    iter_count=0
    #train
    for epoch in range(opts.epoch_count,opts.n_epochs+opts.n_epochs_decay+1):
        for i,data in enumerate(dataset):
            optimizer.zero_grad()
            pred=model(data[0])
            target=data[1].to(device)
            loss = smooth_l1_loss(pred, target)
            loss.backward()
            optimizer.step()
            iter_count+=data[0].size(0)
            if (iter_count%opts.display_freq)==0:
                recorder.record("Epoch/Iter [%d][%d] Loss: %f"%(epoch,iter_count%data_size,loss))
        recorder.record("Finish [%d/%d]"%(epoch,opts.n_epochs+opts.n_epochs_decay))
        model.saveNetwork("latest")
        if(epoch%opts.save_epoch_freq)==0:
            model.saveNetwork(epoch)
        scheduler.step()

    recorder.record("--- Finish ---")
