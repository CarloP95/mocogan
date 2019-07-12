import os
import sys
import skvideo.io
from glob import glob

sys.path.append('../')
from torch_videovision.videotransforms.preprocess import LongestVideo_UCF_101


class Preprocesser():

    def __init__(self, rootDir, transform = None, algorithms = []):

        if len(algorithms) :
            if (not isinstance(algorithms[0], object)):
                raise TypeError(f'Expected algorithm like torch.videovision, of type {type(object)}. Find type {type(algorithms)}')

        self.rootDir              = rootDir
        self.numClasses           = len( glob( os.path.join(rootDir, '*') ) )
        self.videoPaths           = glob( os.path.join(rootDir, '*', '*') )
        self.numVideos            = len(self.videoPaths)
        self.transform            = transform
        self.algorithms           = algorithms

        self.processedVideo = []
        self.toRetainVideo  = []


    def start(self):
        
        prunedPaths = self.videoPaths
        print('Starting processing videos...')
        for algorithm in self.algorithms:
            prunedPaths = algorithm(prunedPaths)

        self.toRetainVideo = prunedPaths


    def getToRetainVideos(self):
        return self.toRetainVideo



def getAlgorithms():

    return \
        [ 
            LongestVideo_UCF_101(),

        ]




if __name__ == '__main__':

    #rootDir     = os.path.join(os.path.dirname(__file__), 'raw_data')
    #videoPaths  = glob( os.path.join(rootDir, '*', '*') ) 

    #from skvideo.measure.viideo import viideo_score

    #videos = [skvideo.io.vread(videoPaths[0], as_grey= True ), skvideo.io.vread(videoPaths[1], as_grey= True)]

    #res = [ viideo_score(videos[0]), viideo_score(videos[1]) ]

    #print(res)

    

    print('Start Preprocessing videos of UCF_101')
    outputFilename = 'Submit_to_model_videos.txt'

    currentPath = os.path.dirname(__file__)
    datasetPath = os.path.join(currentPath, 'raw_data')

    algs = getAlgorithms()

    preprocesser = Preprocesser(datasetPath,  algorithms = algs)
    preprocesser.start()

    toRetain = preprocesser.getToRetainVideos()

    with open(outputFilename, 'w') as file:
        for videopath in toRetain:
            file.write(f'{videopath}\n')

    print('End of Preprocessing.')