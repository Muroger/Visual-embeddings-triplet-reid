import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

import numpy as np
from csv_dataset import CsvDataset
from models.utils import import_from_path
from models import get_model
from csv_dataset import make_dataset_mot
from csv_dataset import make_dataset_default
from csv_dataset import pil_loader_with_crop
import os
import h5py
from argparse import ArgumentParser
import json
import logger as log

def clean_dict(dic):
    """Removes module from keys. This is done when because of DataParallel! TODO what if not DataParallel."""
    fresh_dict = {}
    prefix = "module."
    for key, value in dic.items():
        if key.startswith(prefix):
            key = key[len(prefix):]
        fresh_dict[key] = value
    return fresh_dict


def extract_csv_name(csv_file):
    filename = os.path.basename(csv_file)
    if filename.endswith(".csv"):
        return filename[:-4]
    else:
        return filename


def write_to_h5(output_file, model, endpoints, dataloader, dataset, keys=["emb"]):
    """
    Writes model to h5
    """
    print(len(dataloader), len(dataset))

    print("Model dimension is {}".format(model.module.dim))
    #print('keys', keys)
    if len(keys) == 0:
        raise RuntimeError("Plase specify at least one key that should be written to file.")

    with h5py.File(output_file) as f_out:
        # Dataparallel class!
        datasets = {}
        for key in keys:
            datasets[key] = f_out.create_dataset(key, shape=(len(dataset), ) + model.module.dimensions[key], dtype=np.float32)
        for key in dataset.header:
            datasets[key] = f_out.create_dataset(
                    key,
                    shape=(len(dataset),),
                    dtype=h5py.special_dtype(vlen=str))
        start_idx = 0

        for endpoints, rows in run_forward_pass(dataloader, model, endpoints):
            # TODO this will not work if for some reason some endpoints are shorter than others
            for key in keys:
                end_idx = start_idx + len(endpoints[key])
                datasets[key][start_idx:end_idx] = endpoints[key]
            for key, values in rows.items():
                end_idx = start_idx + len(values)
                datasets[key][start_idx:end_idx] = np.asarray(values)
            start_idx = end_idx
    return output_file

def get_basepath_from_modelpath(path):
    return os.path.dirname(path)

def get_args_path(model_path):
    basepath = get_basepath_from_modelpath(model_path)
    return os.path.join(basepath, "args.json")

class InferenceModel(object):
    def __init__(self, model_path, cuda=True):
        self.cuda = cuda

        args = load_args(model_path)
        self.transform = restore_transform(args)
        model, endpoints = restore_model(args, model_path)
        if self.cuda:
            model = model.cuda()
        self.model = model
        self.model.eval()
        self.endpoints = endpoints

    def __call__(self, images):
        """Forward pass on an image.
        Args:
            data: A PIL image
        """
        data = []
        for image in images:
            data.append(self.transform(image))
        data = torch.cat(data)
        print(data.size())
        if self.cuda:
            data = data.cuda()
        with torch.no_grad():
            self.endpoints = self.model(data, self.endpoints)
        #result = self.endpoints["emb"]
        # mean over crops
        # TODO this depends on the data augmentation
        #self.endpoints["emb"] = result.mean(0)
        # COPY otherwise a reference is passed that will be overwritten 
        # by the next forward pass
        return self.endpoints.copy()


# def restore_transform(args, augmentation):
#     # TODO unify with training routine
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])

#     H = args["image_height"]
#     W = args["image_width"]
#     scale = args["scale"]

#     print(args)
#     to_tensor = transforms.ToTensor()


#     def to_normalized_tensor(crop):
#         return normalize(to_tensor(crop))

#     transform_comp = transforms.Compose([
#             transforms.Resize((int(H*scale), int(W*scale))),
#             augmentation((H, W)),
#             transforms.Lambda(lambda crops: torch.stack([to_normalized_tensor(crop) for crop in crops]))
#         ])

#     return transform_comp

from albumentations import (Compose, Normalize, Resize, CenterCrop)
from albumentations.pytorch import ToTensorV2

def restore_transform(args):
    # TODO unify with training routine

    H = args["image_height"]
    W = args["image_width"]
    scale = args["scale"]

    print(args)

    transform_comp = Compose([
                #Resize(int(H*0.75), int(W*0.75)),
                CenterCrop(int(H*0.75), int(W*0.75)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.)

    return transform_comp

class EmbeddingArch(torch.nn.Module):
    """Simple class to wrap around endpoints and model."""
    def __init__(self, model, endpoints):
        self.model = model
        self.endpoints = endpoints
    def __call__(self, data):
        self.model(data, self.endpoints)



def restore_model(args, model_path):
    basepath = get_basepath_from_modelpath(model_path)
    model, endpoints = import_from_path(basepath, args)
    if model is None:
        model, endpoints = get_model(args)
    #restore trained model
    state_dict = torch.load(model_path)
    state_dict = clean_dict(state_dict)
    model.load_state_dict(state_dict)
    return model, endpoints

def load_args(model_path):
    args_path = get_args_path(model_path)

    with open(args_path, 'r') as f_handle:
        return json.load(f_handle)


def create_embeddings(dataloader, model, endpoints):
    """More of a temporary wrapper, TODO look for cleaner solution."""
    for idx, endpoints in enumerate(run_forward_pass(dataloader, model, endpoints)):
        result = endpoints["emb"]
        #restore batch and crops dimension and use mean over all crops
        print("\rDone (%d/%d)" % (idx, len(dataloader)), flush=True, end='')
        yield  result.data.cpu().numpy()

def run_forward_pass(dataloader, model, endpoints):
    """
    """

    model.eval()

    for idx, (data, _, row) in enumerate(dataloader):
        data = Variable(data).cuda()
        #print(data.size())
        # with cropping there is an additional dimension
        bs, c, h, w = data.size()
        endpoints = model(data.view(-1, c, h, w), endpoints)
        # Transform to original shape
        for key, value in endpoints.items():
            #TODO fix what if not data parallel
            try:
                endpoints[key] = endpoints[key].view((bs,) + model.module.dimensions[key])
                #TODO handle GPU CPU!!!
                endpoints[key] = endpoints[key].data.cpu().numpy()
            except AttributeError:
                continue
                print("Data cannot be reshaped, only return tensors. Skipping {}!".format(key))

        yield endpoints, row
        print("\rDone (%d/%d)" % (idx, len(dataloader)), flush=True, end='')



def run(csv_file, data_dir, model_file, batch_size, crop, make_dataset_func,
         keys, prefix=None, output_dir="embed", overwrite=False):


    experiment = os.path.realpath(model_file).split('/')[-2]
    model_name = os.path.basename(model_file)
    csv_name = extract_csv_name(csv_file)
    output_file = "{}-{}.h5".format(csv_name, model_name)
    if prefix is not None:
        output_file = "{}_{}".format(prefix, output_file)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir, experiment)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    output_file = os.path.join(os.path.abspath(output_dir), output_file)

    if os.path.isfile(output_file):
        if overwrite:
            os.remove(output_file)
            print("Deleted and overwriting file in {}".format(output_file))
        else:
            #TODO create numerated filename
            print("File %s already exists! Please choose a different name." % output_file)
            return output_file
    else:
        print("Creating file in %s" % output_file)

    args = load_args(model_file)
    transform_comp  = restore_transform(args)

    dataset = CsvDataset(csv_file, data_dir, transform=transform_comp, make_dataset_func=make_dataset_func)
    # if crop:
    #     dataset = CsvDataset(csv_file, data_dir, loader_fn=pil_loader_with_crop, transform=transform_comp, make_dataset_func=make_dataset_func)
    # else:
    #     dataset = CsvDataset(csv_file, data_dir, transform=transform_comp, make_dataset_func=make_dataset_func)

    dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size,
                #TODO argument
                num_workers=4,
                pin_memory=True
            )
    model, endpoints = restore_model(args, model_file)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    if keys:
        return write_to_h5(output_file, model, endpoints, dataloader, dataset, keys)
    else:
        return write_to_h5(output_file, model, endpoints, dataloader, dataset)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
            '--output_dir', default="embed",
            help="Output directory for embedding hd5 file."
            )
    parser.add_argument(
            '--filename', default=None,
            help="Output filename")

    parser.add_argument(
            '--csv_file', required=True,
            help="CSV file containing relative paths.")

    parser.add_argument(
            '--data_dir', required=True,
            help="Root dir where the data is stored. This and the paths in the\
            csv file have to result in the correct file path."
            )
    parser.add_argument(
            '--crop', required=False,
            action='store_true',
            help="Crop images on the fly using detection boxes."
            )
    parser.add_argument(
           '--mot', action='store_true')

    parser.add_argument(
            '--model', required=True,
            help="Path to state dict of model."
            )
    parser.add_argument(
            '--prefix', required=False)
    parser.add_argument(
            '--force', default=False, action='store_true',
            help="Overwrite existing file."
            )
    parser.add_argument('--keys', nargs='+', required=False)
    parser.add_argument("--batch_size", default=32, type=int)
    args = parser.parse_args()

    csv_file = os.path.expanduser(args.csv_file)
    data_dir = os.path.expanduser(args.data_dir)
    model_dir = os.path.expanduser(args.model)
    if args.mot == True:
        make_dataset_func = make_dataset_mot
    else:
        make_dataset_func = make_dataset_default
    run(csv_file, data_dir, model_dir, args.batch_size, args.crop, make_dataset_func, 
        args.keys, args.prefix, args.output_dir, args.force)


