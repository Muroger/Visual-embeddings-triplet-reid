import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from csv_dataset import CsvDataset

from triplet_sampler import TripletBatchSampler, TripletBatchWithJunkSampler

from triplet_loss import choices as loss_choices
from triplet_loss import calc_cdist
import sys; sys.path.append('./src/models')
from models import get_model
from models import model_choices
import os
import h5py

from argparse import ArgumentParser

import logger as log
import time


from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

# Lets cuDNN benchmark conv implementations and choose the fastest.
# Only good if sizes stay the same within the main loop!
torch.backends.cudnn.benchmark = True


parser = ArgumentParser()

parser.add_argument('experiment',
        help="Name of the experiment")

parser.add_argument('--output_path', default="./experiments",
        help="Path where logging files are stored.")

parser.add_argument(
        '--csv_file', required=True,
        help="CSV file containing relative paths.")

parser.add_argument(
        '--data_dir', required=True,
        help="Root dir where the data is stored. This and the paths in the\
        csv file have to result in the correct file path."
        )

parser.add_argument(
        '--log_level', default=1, type=int,
        help="logging level"
        )

parser.add_argument(
        '--limit', default=None, type=int,
        help="The maximum number of (Images) that are loaded from the dataset")

parser.add_argument(
        '--P', default=18, type=int,
        help="Number of persons (pids) per batch.")

parser.add_argument(
        '--K', default=4, type=int,
        help="Number of images per pid.")

parser.add_argument(
        '--train_iterations', default=25000, type=int,
        help="Number of training iterations.")

parser.add_argument(
        '--dim', required=True, type=int,
        help="Size of the embedding vector."
        )
parser.add_argument(
        '--decay_start_iteration', default=15000, type=int,
        help="Learningg decay starts at this iteration")

parser.add_argument(
        '--checkpoint_frequency', default=1000, type=int,
        help="After how many iterations a new checkpoint is created.")

parser.add_argument('--margin', default='soft',
        help="What margin to use: a float value, 'soft' for "
        "soft-margin, or no margin if 'none'")

parser.add_argument('--alpha', default=1.0, type=float,
        help="Weight of the softmax loss.")

parser.add_argument('--temp', default=1.0,
        help="Temperature of BatchSoft")

parser.add_argument('--scale', default=1, type=float,
        help="Scaling of images before crop [scale * (image_height, image_width)]")

parser.add_argument('--image_height', default=256, type=int,
        help="Height of image that is fed to network.")

parser.add_argument('--image_width', default=256, type=int,
        help="Width of image that is fed to network.")

parser.add_argument('--lr', default=3e-4, type=float,
        help="Learning rate.")

parser.add_argument('--model', required=True, choices=model_choices)
parser.add_argument('--loss', required=True, choices=loss_choices)
parser.add_argument('--mgn_branches', required=False, nargs='+', type=int,
        help="Branch configuration for mgn network.")
parser.add_argument('--J', type=int,
        help="Number of Junk images sampled.")
parser.add_argument('--restore_checkpoint', type=int,
        help="Checkpoint that is to be restored from existing experiment.")

parser.add_argument('--no_multi_gpu', action='store_true', default=False)

parser.add_argument('--sampler', required=True,
                    choices=["TripletBatchSampler", "TripletBatchWithJunkSampler"])

def extract_csv_name(csv_file):
    filename = os.path.basename(csv_file)
    if filename.endswith(".csv"):
        return filename[:-4]
    else:
        return filename

def adjust_learning_rate(optimizer, t):
    global t0, t1, eps0
    if t <= t0:
        return eps0
    lr = eps0 * pow(0.001, (t - t0) / (t1 - t0))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

num_epochs = 300
def adjust_learning_rate_v2(optimizer, ep):
    start_decay_at_ep = 151
    base_lr = 2e-4
    if ep < start_decay_at_ep:
        return base_lr

    for g in optimizer.param_groups:
        lr = base_lr * (0.001 ** (float(ep + 1 - start_decay_at_ep)
             / (num_epochs + 1 - start_decay_at_ep)))
        g['lr'] = lr
    return lr

#num_epochs = 80
def adjust_learning_rate_v3(optimizer, epoch):
    global t0, t1, eps0
    if epoch <= 40:
        lr = 0.01
    elif epoch <= 60:
        lr = 1e-3
    else:
        lr = 1e-4

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def adjust_alpha_rate(loss, t):
    a1 = 5000
    a2 = 10000
    if t <= a1:
        alpha = 1.0
    elif t < a2:
        alpha = 1.0 - ((t - a1) / (a2 - a1))
    else:
        alpha = 0.0

    loss.a = alpha
    return alpha

def topk(cdist, pids, k):
    """Calculates the top-k accuracy.
    
    Args:
        k: k smallest value
        
    """ 
    batch_size = cdist.size()[0]
    #print('batch_size', batch_size)
    index = torch.topk(cdist, k+1, largest=False, dim=1)[1] #topk returns value and index
    index = index[:, 1:] # drop diagonal

    topk = torch.zeros(cdist.size()[0]).byte()
    topk = topk.cuda()
    topks = []
    for c in index.split(1, dim=1):
        c = c.squeeze() # c is batch_size x 1
        topk = topk | (pids.data == pids[c].data)
        # topk is uint8, this results in a integer division
        acc = torch.sum(topk).double() / batch_size
        topks.append(acc)
    return topks

def var2num(x):
    return x.data.cpu().numpy()

args = parser.parse_args()


csv_file = os.path.expanduser(args.csv_file)
data_dir = os.path.expanduser(args.data_dir)

mod = __import__('triplet_loss')
loss = getattr(mod, args.loss)

# TODO allow arbitrary number of arguments


eps0 = args.lr
t0 = args.decay_start_iteration
t1 = args.train_iterations


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

H = args.image_height
W = args.image_width
scale = args.scale
transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((int(H*scale), int(W*scale))),
        #transforms.RandomCrop((H*0.75, W*0.75)),
        normalize,
        transforms.ToTensor(),
        
    ])

def get_train_transforms():
    return Compose([
            #RandomResizedCrop(int(H*0.75), int(W*0.75)),
            Resize(int(H*0.75), int(W*0.75)),
            #Transpose(p=0.5),
            HorizontalFlip(p=0.2),
            VerticalFlip(p=0.2),
            ShiftScaleRotate(p=0.2),
            #RandomRotate90(p=0.5),
            #HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            #RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)





dataset = CsvDataset(csv_file, data_dir, transform=get_train_transforms(), limit=args.limit)

print("Loaded %d images" % len(dataset))
if args.sampler == "TripletBatchSampler":
    sampler = TripletBatchSampler(args.P, args.K, dataset)
elif args.sampler == "TripletBatchWithJunkSampler":
    sampler = TripletBatchWithJunkSampler(args.P, args.K, args.J, dataset)
else:
    raise RuntimeError("Unknown sampler")


dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=4, pin_memory=True
        )

#also save num_labels

args.num_classes = dataset.num_labels
model, endpoints = get_model(args.__dict__)
log = log.create_logger("h5", args.experiment, args.output_path, args.log_level)
if not args.restore_checkpoint is None:
    from embed import restore_model
    model_path = log.get_model_path(args.restore_checkpoint)
    if model_path:
        model = restore_model(args.__dict__, model_path)
        print("Model was restored from {}.".format(model_path))
        t = args.restore_checkpoint
    else:
        t = 0
else:
    t = 0

import gc
#print(model)    
def memReport():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

#memReport()
def switch_off_running_stats(node):
    for child in node.children():
        switch_off_running_stats(child)
    if type(node) == torch.nn.BatchNorm2d or type(node) == torch.nn.BatchNorm1d:
        node.track_running_stats = False
        print("changed", node)

#switch_off_running_stats(model)
if args.no_multi_gpu:
    model = model.cuda()
else:
    model = torch.nn.DataParallel(model).cuda()
#model = model.cuda()
try:
    margin = float(args.margin)
except ValueError:
    margin = args.margin

#print(model)
loss_param = {"m": margin, "T": args.temp,
              "a": args.alpha, "num_junk_images": args.J}

loss_fn = loss(**loss_param)
optimizer = torch.optim.Adam(model.parameters(), lr=eps0, betas=(0.9, 0.999))
#optimizer = torch.optim.SGD(model.parameters(), lr=eps0, momentum=0.9, weight_decay=5e-4)



training_name = args.experiment + "%s_%s-%s_%d-%d_%f_%d" % (
    extract_csv_name(csv_file), loss_fn.name,
    str(args.margin), args.P,
    args.K, eps0, args.train_iterations)


# new experiment
log.save_args(args)
log.save_description(model)
log.save_model_file(args.model)

# save
# logging
    #emb_dataset = fout.create_dataset("emb", shape=(t1, batch_size,emb_dim), dtype=np.float32)
    #pids_dataset = fout.create_dataset("pids", shape=(t1, batch_size), dtype=np.int)
    #file_dataset = fout.create_dataset("file", shape=(t1, batch_size), dtype=h5py.special_dtype(vlen=str))
    #log_dataset = fout.create_dataset("log", shape=(t1, 6))


print("Starting training: %s" % training_name)
loss_data = {}
#TODO initialize otherwise first batch is wrong
overall_time = time.time()

def calc_junk_acc(logits, targets, threshold=0.5):
    predicted = torch.max(logits, dim=1)
    predicted = predicted[1]
    return torch.sum(targets == predicted).float() / targets.shape[0]

#model.eval()
for epoch in range(num_epochs):
    model = model.train()
    lr = adjust_learning_rate_v2(optimizer, epoch+1)
    for batch_id, (data, target, path) in enumerate(dataloader):
        start_time = time.time()
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target, requires_grad=False)
        endpoints = model(data, endpoints)
#        result.register_hook(lambda x: print("Gradient", x))
        loss_data["dist"] = calc_cdist(endpoints["emb"], endpoints["emb"])
        loss_data["pids"] = target
        loss_data["endpoints"] = endpoints
        #alpha = adjust_alpha_rate(loss_fn, t)
        losses = loss_fn(**loss_data)
        loss_mean = torch.mean(losses)
        topks = topk(loss_data["dist"], target, 5)
        min_loss = float(var2num(torch.min(losses)))
        max_loss =  float(var2num(torch.max(losses)))
        mean_loss = float(var2num(loss_mean))
#        log.write("emb", var2num(endpoints["emb"]), dtype=np.float32)
        log.write("pids", var2num(target), dtype=np.int)
        log.write("file", path, dtype=h5py.special_dtype(vlen=str))
        #log.write("log", [min_loss, mean_loss, max_loss, lr, topks[0], topks[4]], np.float32)
        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()
        #log.write("batch_norm", var2num(model.module.batch_norm.running_mean))
        took = time.time() - start_time
        print("batch {} loss: {:.3f}|{:.3f}|{:.3f} lr: {:.6f} "
              "top1: {:.3f} top5: {:.3f} | took {:.3f}s".format(
            t, min_loss, mean_loss, max_loss, lr,
            topks[0], topks[4], took
            ))
        t += 1
        if t % args.checkpoint_frequency == 0:
            log.save_model_state(model, t)
        if t >= t1:
            break
log.save_model_state(model, t)
log.close()

print("Finished Training! Took: {:.3f}".format(time.time() - overall_time))
