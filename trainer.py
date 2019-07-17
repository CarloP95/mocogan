from torch import nn as nn, optim
from time import sleep, time

import numpy as np
import skvideo.io
import torch
import math
import os

class Trainer(nn.Module):

    def __init__(self, parameters, discriminator_i, discriminator_v, generator, dataloader, transformator,
                 optimParameters = {
                     "dis_i": { "optim": optim.Adam, "lr": 0.0002, "betas": (0.5, 0.999), "weight_decay": 0.00001 }, 
                     "dis_v": { "optim": optim.Adam, "lr": 0.0002, "betas": (0.5, 0.999), "weight_decay": 0.00001 },
                     "gen_i": { "optim": optim.Adam, "lr": 0.0002, "betas": (0.5, 0.999), "weight_decay": 0.00001 }
                     }, 
                 category_criterion = nn.CrossEntropyLoss, gan_criterion = nn.BCEWithLogitsLoss 
                 ):
                     
        super(Trainer, self).__init__()

        self.cuda               = parameters['cuda']
        self.n_epochs           = parameters['n_iter']
        self.interval_log_stat  = parameters['i_log_stat']
        self.soft_labels        = parameters['soft_labels']
        self.random_labels      = parameters['random_labels']
        self.shuffle_labels     = parameters['shuffle_labels']
        self.interval_save      = parameters['i_save_weights']
        self.start_epoch        = parameters['pre_train_epoch']
        self.interval_train_g_d = parameters['i_alternate_train']
        self.image_batch_size   = self.video_batch_size = self.batch_size = parameters['batch_size']

        self.transformator      = transformator                     # Will be used to retrieve stdDev and Medium when saving the videos.
        self.generator          = generator.cuda() if self.cuda else generator
        self.discriminator_i    = discriminator_i.cuda() if self.cuda else discriminator_i
        self.discriminator_v    = discriminator_v.cuda() if self.cuda else discriminator_v

        self.gan_criterion      = gan_criterion().cuda() if self.cuda else gan_criterion()
        self.category_criterion = category_criterion().cuda() if self.cuda else category_criterion()

        self.current_path       = os.path.dirname(__file__)
        self.trained_path       = os.path.join(self.current_path, 'trained_models')
        self.generated_path     = os.path.join(self.current_path, 'generated_videos')

        self.dataloader    = dataloader

        self.wait_between_batches   = 1
        self.wait_between_epochs    = 10
        
        self.optim_discriminator_i  = None
        self.optim_discriminator_v  = None
        self.optim_generator        = None

        self.models = [self.discriminator_i, self.discriminator_v, self.generator]    # Used to loop over models

        index = 0
        for key, internalDictionary in sorted( optimParameters.items() ):
            
            optimizer = internalDictionary["optim"](self.models[index].parameters(), lr = internalDictionary["lr"], betas = internalDictionary["betas"], weight_decay = internalDictionary["weight_decay"])
            
            self.optim_discriminator_i  = optimizer if key == "dis_i" else self.optim_discriminator_i
            self.optim_discriminator_v  = optimizer if key == "dis_v" else self.optim_discriminator_v
            self.optim_generator        = optimizer if key == "gen_i" else self.optim_generator

            index += 1
        
        self.loadState(self.start_epoch)
        # Improved Techniques for Training GANs ## Link: https://arxiv.org/pdf/1606.03498.pdf
        self.trueLabel  = 1 if not self.soft_labels else 0.9
        self.trueLabel  = self.trueLabel + 0.05 if self.random_labels and self.soft_labels else self.trueLabel
        self.falseLabel = 0


    def loadState(self, epoch):
        
        if epoch != 0:

            loadEpoch = epoch
            addString = f"_epoch-{loadEpoch}" if loadEpoch is not None else ""
            
            self.discriminator_i.load_state_dict(torch.load(self.trained_path + f'/Discriminator_I{addString}.model'))
            self.discriminator_v.load_state_dict(torch.load(self.trained_path + f'/VideoDiscriminator{addString}.model'))
            self.generator.load_state_dict(torch.load(self.trained_path + f'/VideoGenerator{addString}.model'))
            self.optim_discriminator_i.load_state_dict(torch.load(self.trained_path + f'/Discriminator_I{addString}.state'))
            self.optim_discriminator_v.load_state_dict(torch.load(self.trained_path + f'/VideoDiscriminator{addString}.state'))
            self.optim_generator.load_state_dict(torch.load(self.trained_path + f'/VideoGenerator{addString}.state'))
        

    def train_discriminator(self, discriminator, sample_true, sample_fake, opt, batch_size, use_categories, shuffle = False):

        opt.zero_grad()

        real_batch = sample_true if isinstance(sample_true, tuple) else sample_true()
        batch = real_batch[0]

        fake_batch, generated_categories = sample_fake(batch_size)

        real_labels, real_categorical = discriminator(batch)
        fake_labels, fake_categorical = discriminator(fake_batch.detach())

        ones  = self.ones_like(real_labels, shuffle)
        zeros = self.zeros_like(fake_labels, shuffle)

        l_discriminator = self.gan_criterion(real_labels, ones) + \
                          self.gan_criterion(fake_labels    , zeros)

        if use_categories:
            # Ask the video discriminator to learn categories from training videos
            categories_gt = torch.squeeze(real_batch[1].long())
            l_discriminator += self.category_criterion(real_categorical.squeeze(), categories_gt)

        l_discriminator.backward()
        opt.step()

        return l_discriminator


    def train_generator(self,
                        image_discriminator, video_discriminator,
                        sample_fake_images, sample_fake_videos,
                        opt, shuffle = False):

        opt.zero_grad()
        image_discriminator.eval()
        video_discriminator.eval()

        # train on images

        fake_batch, generated_categories = sample_fake_images(self.image_batch_size)
        fake_labels, fake_categorical = image_discriminator(fake_batch)
        all_ones = self.ones_like(fake_labels, shuffle)

        l_generator = self.gan_criterion(fake_labels, all_ones)

        # train on videos

        fake_batch, generated_categories = sample_fake_videos(self.video_batch_size)
        fake_labels, fake_categorical = video_discriminator(fake_batch)
        all_ones = self.ones_like(fake_labels, shuffle)

        l_generator += self.gan_criterion(fake_labels, all_ones)
        
        # Ask the generator to generate categories recognizable by the discriminator
        l_generator += self.category_criterion(fake_categorical.squeeze(), generated_categories)

        l_generator.backward()
        opt.step()

        return l_generator    


    def sample_images(self, videos):
        
        batch_size, numChannels, numFrames, height, width = videos.shape
        toReturn = torch.cuda.FloatTensor(size = (batch_size, numChannels, height, width)) if self.cuda else torch.FloatTensor(size = (batch_size, numChannels, height, width))
        
        for idx, video in enumerate(videos):
            randomFrame = np.random.randint(0, video.shape[1])
            toReturn[idx] = video[:, randomFrame, :, :]

        return toReturn, None


    def ones_like(self, tensor, shuffle = False):
        val = self.trueLabel
        toReturnTensor = torch.FloatTensor(tensor.size()).fill_(val) if not self.cuda else torch.cuda.FloatTensor(tensor.size()).fill_(val)

        if self.random_labels:
            toAddNoise = torch.FloatTensor(np.random.uniform(-0.05, 0.05, size = toReturnTensor.size()))
            toAddNoise = toAddNoise.cuda() if self.cuda else toAddNoise
            toReturnTensor += toAddNoise

        if shuffle:
            probs = np.random.uniform(size = toReturnTensor.size(0))
            
            for idx, prob in enumerate(probs):
                toReturnTensor[idx] = self.falseLabel if prob <= 0.05 else toReturnTensor[idx]

        return toReturnTensor


    def zeros_like(self, tensor, shuffle = False):
        val = self.falseLabel

        return torch.FloatTensor(tensor.size()).fill_(val) if not self.cuda else torch.cuda.FloatTensor(tensor.size()).fill_(val)


    @staticmethod
    def timeSince(since):
        now = time()
        s = now - since
        d = math.floor(s / ((60**2)*24))
        h = math.floor(s / (60**2)) - d*24
        m = math.floor(s / 60) - h*60 - d*24*60
        s = s - m*60 - h*(60**2) - d*24*(60**2)
        return '%dd %dh %dm %ds' % (d, h, m, s)


    def checkpoint(self, epoch):

        list_of_models = [(self.discriminator_i, self.optim_discriminator_i), 
                            (self.discriminator_v, self.optim_discriminator_v),
                            (self.generator, self.optim_generator)]

        for model, optimizer in list_of_models:
            filename = os.path.join(self.trained_path, '%s_epoch-%d' % (model.__class__.__name__, epoch))
            torch.save(model.state_dict(), filename + '.model')
            torch.save(optimizer.state_dict(), filename + '.state')


    def save_video(self, fake_video, category, epoch):
        outputdata = (fake_video * self.transformator.stdDev + self.transformator.medium) * 255 #Remove normalization
        outputdata = outputdata.astype(np.uint8)
        file_path = os.path.join(self.generated_path, 'fake_%s_epoch-%d.mp4' % (category.item(), epoch))
        skvideo.io.vwrite(file_path, outputdata)


    def train(self):

        start_time = time()

        l_gen_history = np.zeros(len(self.dataloader)); l_dis_v_history = np.zeros(len(self.dataloader)); l_dis_i_history = np.zeros(len(self.dataloader))
        
        optimizers = [self.optim_discriminator_i, self.optim_discriminator_v, self.optim_generator]

        def sample_fake_image_batch(batch_size):
            return self.generator.sample_images(batch_size)

        def sample_fake_video_batch(batch_size, video_len = None):
            return self.generator.sample_videos(batch_size, video_len= video_len)

        for current_epoch in range(1, self.n_epochs + 1):

            for optimizer in optimizers:
                optimizer.zero_grad()

            shuffleLabels = self.shuffle_labels and current_epoch <= 8
            total_batch = len(self.dataloader)

            try:
                for batch_idx, (real_videos, targets) in enumerate(self.dataloader):

                    for model in self.models:
                        model.train()
                    
                    real_videos = real_videos.cuda() if self.cuda else real_videos
                    # Target range must be between [0, 100], not [1, 101]
                    targets     = targets.cuda() if self.cuda else targets; targets -= 1

                    # Train the discriminators
                    ### Train image discriminator
                    l_image_dis = self.train_discriminator(self.discriminator_i, self.sample_images(real_videos),
                                                        sample_fake_image_batch, self.optim_discriminator_i,
                                                        self.image_batch_size, use_categories=False, shuffle = shuffleLabels)

                    ### Train video discriminator
                    l_video_dis = self.train_discriminator(self.discriminator_v, (real_videos, targets),
                                                        sample_fake_video_batch, self.optim_discriminator_v,
                                                        self.video_batch_size, use_categories= True, shuffle = shuffleLabels)

                    l_gen = 0.0
                    if current_epoch % self.interval_train_g_d == 0:

                        # Train generator
                        l_gen = self.train_generator(self.discriminator_i, self.discriminator_v,
                                                    sample_fake_image_batch, sample_fake_video_batch,  self.optim_generator, shuffle = shuffleLabels)

                    print(f'\rBatch [{batch_idx + 1}/{total_batch}] Loss_Di: {l_image_dis:.4f} Loss_Dv: {l_video_dis:.4f} Loss_Gen: {l_gen:.4f}', end='')
                    
                    l_gen_history[batch_idx] = l_gen;    l_dis_i_history[batch_idx] = l_image_dis;   l_dis_v_history[batch_idx] = l_video_dis
                    sleep(self.wait_between_batches)

            except StopIteration as _:
                continue

            sleep(self.wait_between_epochs)

            if current_epoch % self.interval_log_stat == 0:
                print(f'\n[{current_epoch}/{self.n_epochs}] ({self.timeSince(start_time)}) Loss_Di: {l_dis_i_history.mean():.4f} Loss_Dv: {l_dis_v_history.mean():.4f} Loss_Generator: {l_gen_history.mean():.4f}')
            
            if current_epoch % self.interval_train_g_d == 0:
                self.generator.eval()
                fake_video, generated_category = sample_fake_video_batch(1, np.random.randint(16, 25 * 3))
                self.save_video(fake_video.squeeze().data.cpu().numpy().transpose(1, 2, 3, 0), generated_category,current_epoch)

            if current_epoch % self.interval_save == 0:
                self.checkpoint(current_epoch)