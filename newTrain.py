import os
import torch
from trainer import Trainer
from argparse import ArgumentParser
from dataloading import Transformator, DataLoaderFactory

from models import VideoGenerator, Discriminator_I, VideoDiscriminator, UCF_101
import torch.backends.cudnn as cudnn; cudnn.benchmark = True

def setCLArguments(parser):

    # Enable Cuda Computing
    parser.add_argument('--cuda', type= int, default= 1,
                        help= 'Set to 1 when to use Cuda capable device.')

    # Get Batch Size
    parser.add_argument('--batch_size', type= int, default= 16,
                        help= 'Set Batch size.')

    # Get Num Iteration
    parser.add_argument('--n_iter', type= int, default= 100,
                        help= 'Set Num Epochs to train.')

    # Get Pre Train Epoch
    parser.add_argument('--pre_train', type= int, default= 0,
                        help= 'Set epoch from which load model weights.')

    # Get Interval to Save Models.
    parser.add_argument('--i_save_weights', type= int, default= 1,
                        help= 'Set interval between checkpoint for models.')

    # Get Interval to Log Statistics
    parser.add_argument('--i_log_stat', type= int, default= 1,
                        help= 'Set interval between logging in stdout.')

    # Enable use preprocessing
    parser.add_argument('--preprocessing', type= int, default= 1,
                        help= 'Use Submit_to_model_videos.txt for file list.')

    # Get k to alternate training of discriminators and generator
    parser.add_argument('--i_alternate_train', type= int, default= 2,
                        help= 'Set k epochs in which Generator will not be trained.')

    # Enable Soft Labels
    parser.add_argument('--soft_labels', action= 'store_true', default= False,
                        help= 'Set to use Soft Labels {0.1 : False, 0.9: True}')

    # Enable Soft Random Labels
    parser.add_argument('--random_labels', action= 'store_true', default= False,
                        help= 'Set to use Random Soft Labels.')

    # Enable Shuffle Labels
    parser.add_argument('--shuffle_labels', action= 'store_true', default= False,
                        help= 'Set to shuffle 5/100 labels in first 8 epochs of training.')

    # Enable WGAN, this will disable alternate training.
    parser.add_argument('--i_wasserstein', type= int, default= 0,
                        help= 'Set this to a number different from 0 to turn into a WGAN. Read https://arxiv.org/pdf/1701.07875.pdf')


def getCLArguments(parser):

    args = parser.parse_args()

    cuda                = True if args.cuda and torch.cuda.is_available() else False
    batch_size          = args.batch_size
    soft_labels         = args.soft_labels
    n_iter              = args.n_iter
    i_log_stat          = args.i_log_stat
    pre_train_epoch     = args.pre_train
    i_save_weights      = args.i_save_weights
    i_alternate_train   = args.i_alternate_train
    preprocessing       = args.preprocessing == 1
    random_labels       = args.random_labels
    shuffle_labels      = args.shuffle_labels
    i_wasserstein       = args.i_wasserstein
    
    return {
        "cuda"              : cuda,
        "batch_size"        : batch_size,
        "soft_labels"       : soft_labels,
        "n_iter"            : n_iter, 
        "i_log_stat"        : i_log_stat,
        "pre_train_epoch"   : pre_train_epoch,
        "i_save_weights"    : i_save_weights,
        "i_alternate_train" : i_alternate_train,
        "preprocessed"      : preprocessing,
        "random_labels"     : random_labels,
        "shuffle_labels"    : shuffle_labels,
        "i_wasserstein"     : i_wasserstein
    }


def createModels(parameters):

    n_channels = 3
    dim_z_content=50            #  dimensionality of the content input, ie hidden space [default: 50]
    dim_z_motion=10             #  dimensionality of the motion input [default: 10]
    dim_z_category=101          #  dimensionality of categorical input [default: 101]
    video_lenght = 16           #  hyperparameter T of the mocogan [default: 16]
    cuda = parameters['cuda']

    return Discriminator_I(), VideoDiscriminator(n_channels, dim_z_category, cuda = cuda), \
                 VideoGenerator(n_channels, dim_z_content, dim_z_category, dim_z_motion, video_lenght, cuda)


def getDataloader(batch_size, preprocessed):

    img_size    = 96
    max_frame   = 16            # This prevents cropping the clip into the training algorithm

    t = Transformator(img_size, max_frame)

    factory = DataLoaderFactory(UCF_101, t, preprocessed, batch_size)
    
    return factory.getDataLoader(), t

if __name__ == '__main__':

    parser = ArgumentParser(description= 'Start Training MoCoGAN from Pellegrino source code...')

    setCLArguments(parser)

    clArguments = getCLArguments(parser)

    dis_i, dis_v, gen= createModels(clArguments)

    gen.init_weigths()

    dataloader, transformator = getDataloader(clArguments['batch_size'], clArguments['preprocessed'])

    # Prepare Trainer
    trainer = Trainer(clArguments, dis_i, dis_v, gen, dataloader, transformator)

    try:
        trainer.train()

    except KeyboardInterrupt as _:
        pass