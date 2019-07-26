# coding: utf-8

## Additions for integration with UCF-101
import os
import math
import imageio
import skvideo.io
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from skvideo.io import FFmpegReader
#### End of additions.


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init

class Debug(nn.Module):
    def forward(self, input):
        print(input.shape, flush = True)
        return input


class Noise(nn.Module):
    def __init__(self, use_noise, sigma = 0.2, use_gpu = False):
        super(Noise, self).__init__()
        self.sigma      = sigma
        self.use_gpu    = use_gpu
        self.use_noise  = use_noise
        
        
    def forward(self, arg):
        
        if self.use_gpu and self.use_noise:
            return arg + self.sigma * torch.cuda.FloatTensor(arg.size(), requires_grad = False).normal_()
        
        elif self.use_noise:
            return arg + self.sigma * torch.FloatTensor(arg.size(), requires_grad = False).normal_()
        
        else:
            return arg


# see: _netD in https://github.com/pytorch/examples/blob/master/dcgan/main.py
class Discriminator_I(nn.Module):
    def __init__(self, nc=3, ndf=64, ngpu=1):
        super(Discriminator_I, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 96 x 96
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 48 x 48
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 24 x 24
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 12 x 12
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 6 x 6
            nn.Conv2d(ndf * 8, 1, 6, 1, 0, bias=False),
            # Do not use it, since using BCEWithLogitsLoss nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1), None


class Discriminator_V(nn.Module):
    def __init__(self, nc=3, ndf=64, T=16, ngpu=1, nClasses= 102):
        super(Discriminator_V, self).__init__()
        self.ngpu       = ngpu
        self.nClasses   = nClasses
        
        self.main = nn.Sequential(
            # input is (nc) x T x 96 x 96
            nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x T/2 x 48 x 48
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x T/4 x 24 x 24
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x T/8 x 12 x 12
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x T/16  x 6 x 6
            Flatten(),
            nn.Linear(int((ndf*8)*(T/16)*6*6), 1 + nClasses),
            #nn.Sigmoid()
        )

    def forward(self, input):
    
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            
        else:
            output = self.main(input)
            
        labels = output[:, 1 : output.size(1)]
        output = output[:, 0]
        return output.view(-1, 1).squeeze(1), labels


class VideoDiscriminator(nn.Module):
    def __init__(self, n_channels, n_categories, n_output_neurons=1, use_noise=False, noise_sigma=None, ndf=64, cuda = False):
        super(VideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons + n_categories
        self.use_noise = use_noise
        self.n_categories = n_categories

        self.main = nn.Sequential(
            #Noise(use_noise, sigma=noise_sigma, use_gpu= cuda),
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            #Noise(use_noise, sigma=noise_sigma, use_gpu= cuda),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            #Noise(use_noise, sigma=noise_sigma, use_gpu= cuda),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            #Noise(use_noise, sigma=noise_sigma, use_gpu= cuda),
            nn.Conv3d(ndf * 4, ndf * 8, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(ndf * 8, self.n_output_neurons, 4, 3, 0, bias=False)
        )
    
    def split(self, input):
        return input[:, :input.size(1) - self.n_categories], input[:, input.size(1) - self.n_categories:]

    def forward(self, input):
        h = self.main(input).squeeze()
        labels, categ = self.split(h)
        return labels, categ
        

# see: _netG in https://github.com/pytorch/examples/blob/master/dcgan/main.py
class Generator_I(nn.Module):
    '''
        Constructor
        -----------
        The constructor of Generator_I takes 6 arguments, all optional.
        
        nc:         integer, default= 3
            Num channels of the image to produce.
        
        ngf:        integer, default= 64
            Parameter of the ConvTranspose2d Layers.
        
        nz:         integer, default= 60
            Number of samples for the noise.
            
        ngpu:       integer, default= 1
            Number of GPU on which the model will run.
            
        nClasses:   integer, default= 102
            Number of classes on which the Embedding module will work.
            
        batch_size: integer, default = 16
            Batch size for each argument that will be passed to the model.
            
    
    '''
    
    def __init__(self, nc=3, ngf=64, nz=60, ngpu=1, nClasses= 102, batch_size= 16):
        super(Generator_I, self).__init__()
        self.ngpu = ngpu
        # Addition for Conditioning the Model
        # nClasses = #UCF-101 Action Class + 1 (Fake Class) 
        self.label_sequence = nn.Sequential(
            # labels size [ NumClasses / 16 ]
            nn.Embedding(nClasses, nClasses//16),
            nn.Linear(nClasses//16, nz),
            nn.ReLU(True)
        )
        
        self.combine_sequence = nn.Sequential(
            nn.Linear(ngf*4 + batch_size, ngf*4)
        )
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 6, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 6 x 6
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 12 x 12
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 24 x 24
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 48 x 48
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 96 x 96
        )

    def forward(self, input, labels):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            # Addition to prepare labels to be concatenated with input.
            labels = nn.parallel.data_parallel(self.label_sequence, labels, range(self.ngpu))
            labels = labels.unsqueeze(0).unsqueeze(0)
            labels = labels.transpose(0,2).transpose(1,3)
            
            combinedInput = torch.cat((input, labels), 0).transpose(0,3)
            
            input = nn.parallel.data_parallel(self.combine_sequence, combinedInput, range(self.ngpu)).transpose(3,0)
            
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            
        else:
            labels = self.label_sequence(labels)
            labels = labels.unsqueeze(0).unsqueeze(0)
            labels = labels.transpose(0,2).transpose(1,3)
            
            combinedInput = torch.cat((input, labels), 0).transpose(0,3)
            
            input = self.combine_sequence(combinedInput).transpose(3,0)
            
            output = self.main(input)
            
        return output


class GRU(nn.Module):
    def __init__(self, input_size = 10, hidden_size = 10, gpu=True):
        super(GRU, self).__init__()

        output_size      = input_size
        self._gpu        = gpu
        self.hidden_size = hidden_size

        # define layers
        self.gru    = nn.GRUCell(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.bn     = nn.BatchNorm1d(output_size, affine=False)

    def forward(self, inputs, n_frames):
        '''
        inputs.shape()   => (batch_size, input_size)
        outputs.shape() => (seq_len, batch_size, output_size)
        '''
        outputs = []
        for i in range(n_frames):
            self.hidden = self.gru(inputs, self.hidden)
            inputs = self.linear(self.hidden)
            outputs.append(inputs)
        outputs = [ self.bn(elm) for elm in outputs ]
        outputs = torch.stack(outputs)
        return outputs

    def initWeight(self, init_forget_bias=1):
        # See details in https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
        for name, params in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(params)

            # initialize forget gate bias
            elif 'gru.bias_ih_l' in name:
                b_ir, b_iz, b_in = params.chunk(3, 0)
                init.constant_(b_iz, init_forget_bias)
            elif 'gru.bias_hh_l' in name:
                b_hr, b_hz, b_hn = params.chunk(3, 0)
                init.constant_(b_hz, init_forget_bias)
            else:
                init.constant_(params, 0)

    def initHidden(self, batch_size):
        self.hidden = Variable(torch.zeros(batch_size, self.hidden_size))
        if self._gpu == True:
            self.hidden = self.hidden.cuda()


''' utils '''

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class VideoGenerator(nn.Module):
    def __init__(self, n_channels, dim_z_content, dim_z_category, dim_z_motion,
                 video_length, cuda = False, ngf=64, class_to_idx = None):
        super(VideoGenerator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_category = dim_z_category
        self.dim_z_motion = dim_z_motion
        self.video_length = video_length
        self.gpu = cuda

        self.class_to_idx = class_to_idx

        dim_z = dim_z_motion + dim_z_category + dim_z_content

        self.recurrent = nn.GRUCell(dim_z_motion, dim_z_motion)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_z, ngf * 8, 6, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, self.n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def init_weigths(self, init_forget_bias=1):
        for name, params in self.recurrent.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(params)

            # initialize forget gate bias
            elif 'gru.bias_ih_l' in name:
                b_ir, b_iz, b_in = params.chunk(3, 0)
                init.constant_(b_iz, init_forget_bias)
            elif 'gru.bias_hh_l' in name:
                b_hr, b_hz, b_hn = params.chunk(3, 0)
                init.constant_(b_hz, init_forget_bias)
            else:
                init.constant_(params, 0)


    def sample_z_m(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        h_t = [self.get_gru_initial_state(num_samples)]

        for frame_num in range(video_len):
            e_t = self.get_iteration_noise(num_samples)
            h_t.append(self.recurrent(e_t, h_t[-1]))

        z_m_t = [h_k.view(-1, 1, self.dim_z_motion) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion)

        return z_m

    def sample_z_categ(self, num_samples, video_len, category = None):
        video_len = video_len if video_len is not None else self.video_length

        if self.dim_z_category <= 0:
            return None, np.zeros(num_samples)

        if category:
            classes_to_generate = np.array(category)

        else:
            classes_to_generate = np.random.randint(self.dim_z_category, size=num_samples)

        one_hot = np.zeros((num_samples, self.dim_z_category), dtype=np.float32)
        one_hot[np.arange(num_samples), classes_to_generate] = 1
        one_hot_video = np.repeat(one_hot, video_len, axis=0)

        one_hot_video = torch.from_numpy(one_hot_video)

        if self.gpu:
            one_hot_video = one_hot_video.cuda()

        return Variable(one_hot_video), classes_to_generate

    def sample_z_content(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        content = np.random.normal(0, 1, (num_samples, self.dim_z_content)).astype(np.float32)
        content = np.repeat(content, video_len, axis=0)
        content = torch.from_numpy(content)

        if self.gpu:
            content = content.cuda()

        return Variable(content)

    def sample_z_video(self, num_samples, video_len=None, category = None):
        z_content = self.sample_z_content(num_samples, video_len)
        z_category, z_category_labels = self.sample_z_categ(num_samples, video_len, category)
        z_motion = self.sample_z_m(num_samples, video_len)

        if z_category is not None:
            z = torch.cat([z_content, z_category, z_motion], dim=1)
        else:
            z = torch.cat([z_content, z_motion], dim=1)

        return z, z_category_labels

    def sample_videos(self, num_samples, video_len=None, category = None):
        video_len = video_len if video_len is not None else self.video_length

        z, z_category_labels = self.sample_z_video(num_samples, video_len, category)

        h = self.main(z.view(z.size(0), z.size(1), 1, 1))
        h = h.view( int( h.size(0) / video_len), video_len, self.n_channels, h.size(3), h.size(3))

        z_category_labels = torch.from_numpy(z_category_labels)

        if self.gpu:
            z_category_labels = z_category_labels.cuda()

        h = h.permute(0, 2, 1, 3, 4)
        return h, z_category_labels

    def sample_images(self, num_samples):
        z, z_category_labels = self.sample_z_video(num_samples * self.video_length * 2)

        j = np.sort(np.random.choice(z.size(0), num_samples, replace=False)).astype(np.int64)
        z = z[j, ::]
        z = z.view(z.size(0), z.size(1), 1, 1)
        h = self.main(z)
        return h, None

    def get_gru_initial_state(self, num_samples):
        return Variable(torch.cuda.FloatTensor(num_samples, self.dim_z_motion).normal_()) if self.gpu else Variable(torch.FloatTensor(num_samples, self.dim_z_motion).normal_())

    def get_iteration_noise(self, num_samples):
        return Variable(torch.cuda.FloatTensor(num_samples, self.dim_z_motion).normal_()) if self.gpu else Variable(torch.FloatTensor(num_samples, self.dim_z_motion).normal_())


    def getCorrectClassName(self, classIdx):
        if self.class_to_idx:
            for action, index in self.class_to_idx.items():
                if index == classIdx:
                    return action
        else:
            raise ValueError(f'No dictionary found on this instance of {self.__class__.__name__}')

## Addition for loading target & videoPath from UCF-101 class.
   

def getNumFrames(reader):
    
    try:
        return math.ceil(reader.get_meta_data()['fps'] * reader.get_meta_data()['duration'])
    
    except AttributeError as _:
        filename = reader
        return getNumFrames(imageio.get_reader(filename,  'ffmpeg'))

def readVideoImageio(filename, n_channels= 3):
    
    frames = []
    with imageio.get_reader(filename,  'ffmpeg') as reader:
    
        shape = reader.get_meta_data()['size']
        
        for img in reader:
            try:
                frames.append(img)
                
            except Error as err:
                print(err)
         
    # Paranoid double check
    if not reader.closed:
        reader.close()
        
    videodata = np.array(frames, dtype= np.uint8)
            
    return videodata

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    extensions = list(extensions)
    
    returnValue = False
    for extension in extensions:
        returnValue = returnValue or filename.lower().endswith(extension)
    
    return returnValue


def make_dataset(dir, class_to_idx, extensions=None, classes = []):
    videos = []

    if os.path.isdir(dir):

        dir = os.path.expanduser(dir)
        
        if extensions is not None:
            def is_valid_file(x):
                return has_file_allowed_extension(x, extensions)
            
        for target in sorted(class_to_idx.keys()):
            
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                if len(classes) > 0:
                    if not any([element.lower() in root.lower() for element in classes]):
                        continue

                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = (path, class_to_idx[target])
                        videos.append(item)

    else: # It is a file containing preprocessed informations.
        with open(dir, 'r') as file:
            for line in file:
                if line == '':
                    continue
                line      = line.rstrip('\n\r')
                target    = os.path.split( os.path.split(line)[0] ) [1]
                path      = os.path.join( os.path.split(dir)[0], line)
                item      = (path, class_to_idx[target])
                videos.append(item)
                    
    return videos


## Addition for loading videos avoiding to get Out Of Memory

class UCF_101(Dataset):
    """
        Summary
        ----------
            A class that extends the abstract Dataset class to load lazily videos from disk.

        Parameters
        ----------

        rootDir: string
            Absolute path to the directory in which the subfolders of UCF-101 (named with "Action") are found.

        videoHandler: module
            Hook to add new module to handles video loading.
            
        supportedExtensions: List of Strings
            A list of extensions to load. E.g. ["mp4", "avi"]
            
        transform: torchvision.transforms.Compose
            Sequence of transformation to apply to the dataset while loading.
            
        Constructor:
        ----------
            It requires that the in the previous directory with respect to @Param rootDir it can find the directory ucfTrainTestList
            where it can read the file named classInd.txt where the mapping "Target" "Index" can be loaded.
            
        Attributes:
        ----------
            videoLengths: 
                A dictionary that contains as key the filepath and as value the nframes of the video.
                This is populated in a lazy way, every time that a video is loaded for the first time into memory.
        
    """    
    
    def __init__(self, rootDir, dictClassDir = '', videoHandler = readVideoImageio, supportedExtensions= [], transform= None, classes = []):
        
        ucfDictFilename = "classInd.txt"                    #Used to load the file classes.
        ucfTrainTestDirname = "ucfTrainTestlist"            #Used to find the class file.
        
        if dictClassDir == '':
            previousDir = [*(os.path.split(rootDir)[:-1])][0]
            self.dictPath = os.path.join(previousDir, ucfTrainTestDirname, ucfDictFilename)
        
        else:
            self.dictPath = dictClassDir
            
        self.currentDir     = os.path.dirname(__file__)
        self.rootDir        = os.path.join(self.currentDir, rootDir)
        self.videoHandler   = videoHandler
        self.transform      = transform
        self.videoLengths   = {}
        self.classes        = classes
        
        self.class_to_idx   = self.loadDict(self.dictPath)
        
        self.samples        = make_dataset(self.rootDir, self.class_to_idx, supportedExtensions, classes = classes)
        

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        
        path, target = self.samples[index]
        
        #readVideo = self.videoHandler(path, verbosity= 1)
        #readVideo = self.videoHandler(path, verbosity= 1)
        
        
        #inputdict = {"-threads": "4", "-s": "96x96", "-pix_fmt" : "yuv420p"}
        
        #with open(path, "r") as _:
            #with FFmpegReader(path, inputdict= inputdict, verbosity= 0) as reader:
                
                #T, M, N, C = reader.getShape()

                #readVideo = np.empty((T, M, N, C), dtype=reader.dtype)
                #for idx, frame in enumerate(reader.nextFrame()):
                    #readVideo[idx, :, :, :] = frame
        
        
        readVideo = self.videoHandler(path)
        
        
        self.videoLengths[path] = readVideo.shape[0] #getNumFrames(path)
        
        if self.transform:
            readVideo = self.transform(readVideo)
        
        return readVideo, target


    def loadDict(self, filepath):

        dictClassesIdx = {}

        try:
            assert self.classes[0]
            dictClassesIdx = {self.classes[idx] : idx + 1 for idx in range(len(self.classes))}

        except:

            try:
                with open(filepath) as file:
                    for line in file:
                        dictClassesIdx[ line.split() [1]] = int( line.split() [0] )
                    
            except IsADirectoryError as _:
                classes = glob(os.path.join(filepath, '*'))
                assert classes[0]
                classes = [os.path.split(action)[1] for action in classes]
                dictClassesIdx = {classes[idx] : idx + 1 for idx in range(len(classes))}

            except FileNotFoundError as error:
                print(error)

        return dictClassesIdx
#### End of Addition.
