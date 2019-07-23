##########################################################################################
## MIT License                                                                          ##
##                                                                                      ##
## Copyright (c) [2019] [ CarloP95 carlop95@hotmail.it,                                 ##
##                        vitomessi vitomessi93@gmail.com ]                             ##
##                                                                                      ##
##                                                                                      ##
## Permission is hereby granted, free of charge, to any person obtaining a copy         ##
## of this software and associated documentation files (the "Software"), to deal        ##
## in the Software without restriction, including without limitation the rights         ##
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell            ##
## copies of the Software, and to permit persons to whom the Software is                ##
## furnished to do so, subject to the following conditions:                             ##
##                                                                                      ##
## The above copyright notice and this permission notice shall be included in all       ##
## copies or substantial portions of the Software.                                      ##
##                                                                                      ##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR           ##
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,             ##
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE          ##
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER               ##
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,        ##
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE        ##
## SOFTWARE.                                                                            ##
##########################################################################################

#### Additions for integration with UCF-101

from functools import reduce
from operator import mul
from tqdm import tqdm

import numpy as np
import torch


class StatisticsCalculator:
    def __init__(self, dataloader, calculate_std = True, mean = None):
        self.calculate_std  = calculate_std
        self.dataloader     = dataloader
        self.medium         = mean

        for element, _ in self.dataloader:
            element = element[0]
            self.numChannels = element.shape[-1]
            self.dimensionToReduce = [*range(len(element.shape) - 1)]
            break

    
    def start(self):
        # Using DoubleTensor to avoid precision errors.
        accumulator     = torch.DoubleTensor(self.numChannels).fill_(0)
        totalNumEls     = 0
        medium          = torch.DoubleTensor(self.medium) if self.medium is not None else None
        calculateNumEls = True if medium is not None else False
        std             = None

        if medium is None:

            print(f'Starting routine for Mean values in dimension {self.dimensionToReduce} of each element.')
            for element, _ in tqdm(self.dataloader):
                
                # Reduce batch dimensions
                element         = element[0]
                accumulator     += element.sum(self.dimensionToReduce).double()
                totalNumEls     += reduce(mul, [element.shape[dim] for dim in self.dimensionToReduce])

            medium = (accumulator/totalNumEls)
            print(f'Finded medium values: {medium.tolist()}')

        if self.calculate_std:

            accumulator = torch.DoubleTensor(self.numChannels).fill_(0)

            print(f'\n\nStarting routine for Std deviation in dimension {self.dimensionToReduce} of each element.')
            for element, _ in tqdm(self.dataloader):
            
                # Reduce batch dimensions
                element         = element[0].double()
                accumulator     += ((element - medium)**2).sum(self.dimensionToReduce).double()
                if calculateNumEls:
                    totalNumEls     += reduce(mul, [element.shape[dim] for dim in self.dimensionToReduce])

            toSqrt  = accumulator/totalNumEls
            std     = toSqrt.sqrt().tolist()


        return medium.tolist(), std


        

####
