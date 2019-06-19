# coding: utf-8

#### Additions for integration with UCF-101

import numpy as np
import torch

class _MeanCalculator():
    
    def __init__(self):
        
        self.totalPixels    = 0
        self.mean           = None
        
        
    def executeMean(self, values, totalPixels = 0):
        
        if not isinstance(values, np.ndarray):
            raise TypeError(f"Values must be of type {type(np.ndarray)}. Find {type(values)}")
        
        return values / totalPixels
    
    
    def getToReduceDimensions(self, video, n_channels):
        
        videoSize = video.shape
        
        if n_channels in videoSize or 1 in videoSize:
            
            dimensionsToReduce = []
            
            for idx, dim in enumerate(videoSize):
                
                if dim == n_channels or dim == 1:
                    continue
                
                dimensionsToReduce.append(idx)
            
            return tuple(dimensionsToReduce)
        
        else:
            raise TypeError(f"Can't get num_channels: must be 1 or 3, find shape {videoSize}")


    def reduceChannels(self, video, n_channels = 3):
        
        if not isinstance(video, np.ndarray):
            raise TypeError(f"reduceChannels method is implemented only for {type(np.ndarray)}, can't convert {type(video)}")
        
        videoSize = video.shape
        reducedArray = None
        
        dimensionsToReduce = self.getToReduceDimensions(video, n_channels)
        
        # ### Increment self attribute to self calculate mean
        self.totalPixels = self.totalPixels + ((videoSize[1] * videoSize[2]) * videoSize[0])
        
        return np.add.reduce(video, tuple(dimensionsToReduce))


class StatisticsCalculator():
    
    def __init__(self):
        
        self.std        = None
        self.mean       = _MeanCalculator()
        self.meanValue  = 0
       
       
    def reduceChannels(self, video, n_channels = 3):
        
        return self.mean.reduceChannels(video, n_channels)
    
    
    def executeMean(self, values, totalPixels):
        
        return self.mean.executeMean(values, totalPixels)


    def accumulateForStd(self, video, mean = 0):
        
        subtracted = video - mean
        
        dimensionsToReduce = self.mean.getToReduceDimensions(video, 3)
        
        return np.add.reduce(subtracted**2, dimensionsToReduce)
        

    def getStatistics(self, dataloader):
        
        reducedChannels = np.zeros((3))
        
        # ### Execute Mean
        for object, _ in dataloader:
            
            if isinstance(object, torch.Tensor):
                object = object.numpy()
                
            reducedChannels = reducedChannels + self.reduceChannels(object)
        
        self.meanValue = self.executeMean(reducedChannels, self.mean.totalPixels)
        
        print(f"Mean is: {self.meanValue}")
        
        # ### Execute Std deviation
        
        # ### ### Execute summation
        summation = np.zeros((3))
        
        for object, _ in dataloader:
            if isinstance(object, torch.Tensor):
                object = object.numpy()
                
            summation = summation + self.accumulateForStd(object, self.meanValue)
            
        # ### ### Divide and take squared root
        self.std = np.sqrt( summation / self.mean.totalPixels )
        
        print(f"Std is: {self.std}")
        
        return self.meanValue, self.std

####
