
from __future__ import print_function
from math import ceil
from os.path import join, exists
from os import makedirs
import logging
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import h5py
import faiss
import numpy as np
import parser
import commons
import datasets_ws
import network
from datetime import datetime
import multiprocessing
import torch.nn as nn

# Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.output_folder = join("runs", args.exp_name,
                          start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.output_folder)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")
logging.info(
    f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

# Creation of Datasets
logging.debug(f"Loading dataset Pitts30k from folder {args.datasets_folder}")



dataset = datasets_ws.BaseDataset(args, args.datasets_folder, "pitts30k", "train")

def get_clusters(cluster_set, model, encoder_dim=256):
    args = parser.parse_arguments()
    nDescriptors = 50000
    nPerImage = 100
    nIm = ceil(nDescriptors/nPerImage)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=args.num_workers, batch_size=args.infer_batch_size, shuffle=False, 
                pin_memory=args.device,
                sampler=sampler)

    if not exists(join(args.datasets_folder, 'centroids')):
        makedirs(join(args.datasets_folder, 'centroids'))

    initcache = join(args.datasets_folder, 'centroids_' + str(args.netvlad_clusters) + '_' + str(args.backbone) + '_desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors", 
                        [nDescriptors, encoder_dim], 
                        dtype=np.float32)

            for iteration, (input, indices) in enumerate(data_loader, 1):
                input = input.to(args.device)
                image_descriptors = model(input).view(input.size(0), encoder_dim, -1).permute(0, 2, 1)

                batchix = (iteration-1)*args.infer_batch_size*nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix*nPerImage
                    dbFeat[startix:startix+nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration, 
                        ceil(nIm/args.infer_batch_size)), flush=True)
                del input, image_descriptors
        
        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(encoder_dim, args.netvlad_clusters, niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')

# Initialize model
model =  nn.Sequential(network.get_backbone(args), network.L2Norm())
model.to(args.device)

get_clusters(dataset, model)