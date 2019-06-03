# coding: utf-8

import os
import argparse
import glob
import time
import math
import skvideo.io
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Lambda, Compose

from models import Discriminator_I, Discriminator_V, Generator_I, GRU, UCF_101


parser = argparse.ArgumentParser(description='Start trainning MoCoGAN.....')
parser.add_argument('--cuda', type=int, default=1,
                     help='set -1 when you use cpu')
parser.add_argument('--ngpu', type=int, default=1,
                     help='set the number of gpu you use')
parser.add_argument('--batch-size', type=int, default=16,
                     help='set batch_size, default: 16')
parser.add_argument('--niter', type=int, default=120000,
                     help='set num of iterations, default: 120000')
parser.add_argument('--pre-train', type=int, default=-1,
                     help='set 1 when you use pre-trained models')

## Additions for training on UCF-101
EXPLORATORY_DATA_ANALYSIS = False

parser.add_argument('--i_epochs_checkpoint', type=int, default=10,
                     help='set num of epochs between checkpoints, default: 10')
parser.add_argument('--i_epochs_saveV', type=int, default=100,
                     help='set num of epochs between save fake video, default: 10')
parser.add_argument('--i_epochs_display', type=int, default=10,
                     help='set num of epochs between print information, default: 1')
#### End of additions for UCF-101

args       = parser.parse_args()
cuda       = args.cuda
ngpu       = args.ngpu
batch_size = args.batch_size
n_iter     = args.niter
pre_train  = args.pre_train

## Addition for training on UCF-101
n_epochs_saveV      = args.i_epochs_saveV
n_epochs_display    = args.i_epochs_display
n_epochs_check      = args.i_epochs_checkpoint
max_frame           = 30
#### End of additions

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
if cuda == True:
    torch.cuda.set_device(0)


''' prepare dataset '''

current_path = os.path.dirname(__file__)
resized_path = os.path.join(current_path, 'resized_data')
files = glob.glob(resized_path+'/*/*')

#transformation = Compose([ToTensor(), Lambda(lambda tensor: (tensor - tensor.mean() )/ tensor.std())])

transformation = Compose([Lambda(lambda video: video.transpose(3, 0, 1, 2)/255.0),
                            Lambda(lambda video: video[ : , : max_frame, :, : ]),
                            Lambda(lambda video: torch.FloatTensor(video))])

filenameDictClassesIdx = "classInd.txt"
dictClassesIdx = {}

with open(os.path.join(current_path, "ucfTrainTestlist", filenameDictClassesIdx)) as file:
    for line in file:
        dictClassesIdx[ line.split() [1]] = int( line.split() [0] )

dataset = DatasetFolder(resized_path, skvideo.io.vread, ["mp4"], transform= transformation)
dataset.class_to_idx = dictClassesIdx


dataloader = DataLoader(dataset, batch_size= batch_size, shuffle= True, num_workers= 4, pin_memory= True, drop_last= True)


if (EXPLORATORY_DATA_ANALYSIS):

    minimum = 12000
    #for line in files:
        #print (f"Started loading one of {len(files)} videos into memory... It will be fast.")
    video = skvideo.io.vread(files[0])
    print(str(video.shape))
    video = video.transpose(3, 0, 1 ,2) / 255.0
    video = torch.FloatTensor(video)
    print(str(video.shape))
    #print (f"Ended transforming into tensor: size is { (video.nelement() * video.element_size()) / (1024 * 1024)} MB")
    #print (f"Final memory would be about {(video.nelement() * video.element_size()) / (1024 * 1024 * 1024) * len(files)} GB")
    # transpose each video to (nc, n_frames, img_size, img_size), and devide by 255
    if (video.shape[0] < minimum ):
        minimum = video.shape[0] #Minimum is 180


    print(minimum)

    #    video = video.transpose(3, 0, 1, 2) / 255.0



''' prepare video sampling '''

n_videos = len(dataset) #len(videos)
T = 16

# for true video
def trim(video):
    start = np.random.randint(0, video.shape[1] - (T+1))
    end = start + T
    return video[:, start:end, :, :]

# for input noises to generate fake video
# note that noises are trimmed randomly from n_frames to T for efficiency
def trim_noise(noise):
    #print("-----TRIMMING NOISE-----")
    #print(f"Noise Size: {noise.size()}")
    start = np.random.randint(0, noise.size(1) - (T+1))
    end = start + T
    #print("-----END OF TRIMMING NOISE-----")
    return noise[:, start:end, :, :, :]

def random_choice():
    X = []
    for _ in range(batch_size):
        video = videos[np.random.randint(0, n_videos-1)]
        video = torch.Tensor(trim(video))
        X.append(video)
    X = torch.stack(X)
    return X



# video length distribution
#video_lengths = [video.shape[1] for video in videos]


''' set models '''

img_size = 96
nc = 3
ndf = 64 # from dcgan
ngf = 64
d_E = 10
hidden_size = 100 # guess
d_C = 50
d_M = d_E
nz  = d_C + d_M
criterion = nn.BCELoss()

dis_i = Discriminator_I(nc, ndf, ngpu=ngpu)
dis_v = Discriminator_V(nc, ndf, T=T, ngpu=ngpu)
gen_i = Generator_I(nc, ngf, nz, ngpu=ngpu)
gru = GRU(d_E, hidden_size, gpu=cuda)
gru.initWeight()


''' prepare for train '''

label = torch.FloatTensor()

def timeSince(since):
    now = time.time()
    s = now - since
    d = math.floor(s / ((60**2)*24))
    h = math.floor(s / (60**2)) - d*24
    m = math.floor(s / 60) - h*60 - d*24*60
    s = s - m*60 - h*(60**2) - d*24*(60**2)
    return '%dd %dh %dm %ds' % (d, h, m, s)

trained_path = os.path.join(current_path, 'trained_models')
def checkpoint(model, optimizer, epoch):
    filename = os.path.join(trained_path, '%s_epoch-%d' % (model.__class__.__name__, epoch))
    torch.save(model.state_dict(), filename + '.model')
    torch.save(optimizer.state_dict(), filename + '.state')

def save_video(fake_video, epoch):
    outputdata = fake_video * 255
    outputdata = outputdata.astype(np.uint8)
    dir_path = os.path.join(current_path, 'generated_videos')
    file_path = os.path.join(dir_path, 'fakeVideo_epoch-%d.mp4' % epoch)
    skvideo.io.vwrite(file_path, outputdata)


''' adjust to cuda '''

if cuda == True:
    dis_i.cuda()
    dis_v.cuda()
    gen_i.cuda()
    gru.cuda()
    criterion.cuda()
    label = label.cuda()


# setup optimizer
lr = 0.0002
betas=(0.5, 0.999)
optim_Di  = optim.Adam(dis_i.parameters(), lr=lr, betas=betas)
optim_Dv  = optim.Adam(dis_v.parameters(), lr=lr, betas=betas)
optim_Gi  = optim.Adam(gen_i.parameters(), lr=lr, betas=betas)
optim_GRU = optim.Adam(gru.parameters(),   lr=lr, betas=betas)


''' use pre-trained models '''

if pre_train == True:
    dis_i.load_state_dict(torch.load(trained_path + '/Discriminator_I.model'))
    dis_v.load_state_dict(torch.load(trained_path + '/Discriminator_V.model'))
    gen_i.load_state_dict(torch.load(trained_path + '/Generator_I.model'))
    gru.load_state_dict(torch.load(trained_path + '/GRU.model'))
    optim_Di.load_state_dict(torch.load(trained_path + '/Discriminator_I.state'))
    optim_Dv.load_state_dict(torch.load(trained_path + '/Discriminator_V.state'))
    optim_Gi.load_state_dict(torch.load(trained_path + '/Generator_I.state'))
    optim_GRU.load_state_dict(torch.load(trained_path + '/GRU.state'))


''' calc grad of models '''

def bp_i(inputs, y, retain=False):
    label.resize_(inputs.size(0)).fill_(y)
    labelv = Variable(label)
    outputs = dis_i(inputs)
    err = criterion(outputs, labelv)
    err.backward(retain_graph=retain)
    toReturnErr = err.data[0] if err.size() == torch.Tensor().size() else err.item()
    return toReturnErr, outputs.data.mean()

def bp_v(inputs, y, retain=False):
    #print("----BackPropagate_V-----")
    #print(inputs.size())
    label.resize_(inputs.size(0)).fill_(y)
    labelv = Variable(label)
    outputs = dis_v(inputs)
    err = criterion(outputs, labelv)
    err.backward(retain_graph=retain)
    toReturnErr = err.data[0] if err.size() == torch.Tensor().size() else err.item()
    #print("----End of BackPropagate_V-----")
    return toReturnErr, outputs.data.mean()


''' gen input noise for fake video '''

def gen_z(n_frames, batch_size = batch_size):
    #print("----Generating Z-----")
    #print(f"N_FRAMES: {n_frames}")
    #print(f"BATCH_SIZE: {batch_size}")
    #print(f"D_C: {d_C}")
    #print(f"D_E: {d_E}")
    #print(f"nz: {nz}")
    z_C = Variable(torch.randn(batch_size, d_C))
    #  repeat z_C to (batch_size, n_frames, d_C)
    z_C = z_C.unsqueeze(1).repeat(1, n_frames, 1)
    eps = Variable(torch.randn(batch_size, d_E))
    if cuda == True:
        z_C, eps = z_C.cuda(), eps.cuda()

    gru.initHidden(batch_size)
    # notice that 1st dim of gru outputs is seq_len, 2nd is batch_size
    z_M = gru(eps, n_frames).transpose(1, 0)
    z = torch.cat((z_M, z_C), 2)  # z.size() => (batch_size, n_frames, nz)
    #print("----End Generating Z-----")
    return z.view(batch_size, n_frames, nz, 1, 1)


''' train models '''

start_time = time.time()

print(f"Starting training: CUDA is { 'On' if cuda == True else 'Off'}")

for epoch in range(1, n_iter+1):
    ''' prepare real images '''
    # real_videos.size() => (batch_size, nc, T, img_size, img_size)

    # Get data iterator
    data_iter = iter(dataloader) #Iterator
    data_len = len(dataloader) #Num Batches
    data_i = 0

    processedClass = None

    while data_i < data_len:

        try:
            batch_size = 16
            (real_videos, labels) = next(data_iter) #random_choice()

            ''' Process 1 video for each class while testing. '''
            #if (labels in processedClass):
            #    continue

            #else:
            #    processedClass.append(labels)
            ''' Process only 1 video class'''
            #if processedClass is None:
            #    processedClass = labels.item()
            #else:
            #    if processedClass != labels.item():
            #        continue

            for (key, val) in dictClassesIdx.items():
                if ( val in labels.tolist() ):
                    pass
                    #print(key)

            if cuda == True:
                real_videos = real_videos.cuda()

            real_videos = Variable(real_videos)
            real_img = real_videos[:, :, np.random.randint(0, T), :, :]

            ''' prepare fake images '''
            # note that n_frames is sampled from video length distribution
            n_frames = T + 2 + np.random.randint(0, real_videos.size()[2]) #video_lengths[np.random.randint(0, n_videos)]
            Z = gen_z(n_frames, batch_size)  # Z.size() => (batch_size, n_frames, nz, 1, 1)
            # trim => (batch_size, T, nz, 1, 1)
            Z = trim_noise(Z)
            # generate videos
            Z = Z.contiguous().view(batch_size*T, nz, 1, 1)
            fake_videos = gen_i(Z)
            fake_videos = fake_videos.view(batch_size, T, nc, img_size, img_size)
            # transpose => (batch_size, nc, T, img_size, img_size)
            fake_videos = fake_videos.transpose(2, 1)
            # img sampling
            fake_img = fake_videos[:, :, np.random.randint(0, T), :, :]

            ''' train discriminators '''
            # video
            dis_v.zero_grad()
            randomStartFrameIdx = np.random.randint(0, real_videos.size()[2] - T - 1)
            #print("-----INFOS-----")
            #print(f"RandomStartFrame:{randomStartFrameIdx}")
            #print(f"Video Size:{real_videos.size()}")
            #print("-----END OF INFOS-----")
            croppedRealVideos = real_videos[:,:,randomStartFrameIdx: randomStartFrameIdx + T, :, :]
            err_Dv_real, Dv_real_mean = bp_v(croppedRealVideos, 0.9)
            err_Dv_fake, Dv_fake_mean = bp_v(fake_videos.detach(), 0)
            err_Dv = err_Dv_real + err_Dv_fake
            optim_Dv.step()
            # image
            dis_i.zero_grad()
            err_Di_real, Di_real_mean = bp_i(real_img, 0.9)
            err_Di_fake, Di_fake_mean = bp_i(fake_img.detach(), 0)
            err_Di = err_Di_real + err_Di_fake
            optim_Di.step()


            ''' train generators '''
            gen_i.zero_grad()
            gru.zero_grad()
            # video. notice retain=True for back prop twice
            err_Gv, _ = bp_v(fake_videos, 0.9, retain=True)
            # images
            err_Gi, _ = bp_i(fake_img, 0.9)
            optim_Gi.step()
            optim_GRU.step()

            '''Increment index for Batch'''
            data_i = data_i + 1
        except StopIteration:
            break
            ani
        except KeyboardInterrupt:
            save_video(fake_videos[0].data.cpu().numpy().transpose(1, 2, 3, 0), epoch)
            checkpoint(dis_i, optim_Di, epoch)
            checkpoint(dis_v, optim_Dv, epoch)
            checkpoint(gen_i, optim_Gi, epoch)
            checkpoint(gru,   optim_GRU, epoch)


    if epoch % n_epochs_display == 0:
        print('[%d/%d] (%s) Loss_Di: %.4f Loss_Dv: %.4f Loss_Gi: %.4f Loss_Gv: %.4f Di_real_mean %.4f Di_fake_mean %.4f Dv_real_mean %.4f Dv_fake_mean %.4f'
              % (epoch, n_iter, timeSince(start_time), err_Di, err_Dv, err_Gi, err_Gv, Di_real_mean, Di_fake_mean, Dv_real_mean, Dv_fake_mean))

    if epoch % n_epochs_saveV == 0:
        save_video(fake_videos[0].data.cpu().numpy().transpose(1, 2, 3, 0), epoch)

    if epoch % n_epochs_check == 0:
        checkpoint(dis_i, optim_Di, epoch)
        checkpoint(dis_v, optim_Dv, epoch)
        checkpoint(gen_i, optim_Gi, epoch)
        checkpoint(gru,   optim_GRU, epoch)
