import os
import sys

from models import UCF_101
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

sys.path.append('../')
from torch_videovision.videotransforms.tensor_transforms import Normalize, SpatialRandomCrop
from torch_videovision.videotransforms.volume_transforms import ToTensor, ClipToTensor, TransposeChannels
from torch_videovision.videotransforms.video_transforms import ColorJitter, RandomTemporalCrop, RandomHorizontalFlip, RandomCrop, Resize


def getPaths(preprocessed = False):
    """
    Function used to get the dataset directories and correct mapping for (classes || targets)
    """
    current_path = os.path.dirname(__file__)
    return os.path.join(current_path, "raw_data") if not preprocessed else os.path.join(current_path, "Submit_to_model_videos.txt"), \
                os.path.join(current_path, "ucfTrainTestlist", "classInd.txt")


class Transformator():
    """
    Constructor:
        img_size, type = int
            Size to which image will be resized.

        max_frame, type = int
            Max number of frames for the taken videos.

        interpolation, type = String, default = bilinear
            Interpolation type for the Resize Transformation. Bilinear is slower but gives better results.
        
        brightness, type = Float, default = 0.2
        contrast,  type = Float, default = 0.7
        saturation, type = Float, default = 0.2
        hue, type = Float, default = 0.2
            Parameters used for the ColorJitter transformation. They are randomized on the basis of the range.
            Keep these parameters small.
        
        medium, type = Float, default = 0.5
        stdDev, type = Float, default = 0.5
            Parameters used for the Normalization of data.
        
        crop_size, type = Tuple, default ( img_size * 2, img_size * 3 )
            Parameters used for SpatialRandomCrop
        
    """
    def __init__(self, img_size, max_frame, interpolation = 'bilinear', # 'nearest'
                    brightness = 0.2, contrast = 0.7, saturation = 0.5, hue = 0.1,
                    medium = 0.5, stdDev = 0.5, crop_size = (None, None)
                ):

        self.img_size       = img_size
        self.max_frame      = max_frame
        self.interpolation  = interpolation
        self.medium         = medium;       self.stdDev   = stdDev
        self.brightness     = brightness;   self.contrast = contrast;   self.saturation = saturation;    self.hue = hue

        self.crop_height  = crop_size[0] if crop_size[0] is not None else img_size * 2
        self.crop_width   = crop_size[1] if crop_size[1] is not None else img_size * 3
    

    """
    Main method to be used to get the Compose Transformation for perform data augmentation and preprocessing.
    """
    def getComposeTransformation(self):

        return Compose([    
                            TransposeChannels(),
                            RandomTemporalCrop(self.max_frame),
                            SpatialRandomCrop( (self.crop_height, self.crop_width) ),
                            RandomHorizontalFlip(),
                            TransposeChannels(reverse = True),
                            Resize((self.img_size, self.img_size), interpolation = self.interpolation),
                            ColorJitter(brightness = self.brightness, contrast = self.contrast, saturation = self.saturation, hue = self.hue),
                            ClipToTensor(div_255 = True),
                            Normalize( self.medium, self.stdDev)
                        ])


class DataLoaderFactory():
    """
    Constructor:
        datasetModel, type = UCF_101
            UCF_101 class to load videos from file system in a lazy way.
        
        transformator, type = Transformator || Compose
            Used to get transformation to preprocess data.
        
        preprocessed, type = bool
            Used to choose between preprocessed videos and raw ones.

        batch_size, type = int, default = 16
            Used to decide how much video to load and process in each step.

        num_workers, type = int, default = 8
            Num workers that will be used to load from File System.
        
        extensions, type = list, default = ['avi']
            Extensions supported to load videos. 
    """
    def __init__(self, datasetModel, transformator, preprocessed, batch_size = 16, num_workers = 8, extensions = ["avi"]):
        
        # TypeCheck for transformator parameter
        if not isinstance(transformator, Transformator) and not isinstance(transformator, Compose):
            raise TypeError("Method __init__ of DataLoaderFactory used with transformator parameters not of instance dataloading.Transformator. Use predefined class dataloading.Transformator as creator of Compose or pass a Compose.")

        # TypeCheck for datasetModel parameter
        if not type(datasetModel) == type(UCF_101):
            raise NotImplementedError("Method __init__ of DataLoaderFactory used with datasetModel different from predefined UCF_101. Use predefined models.UCF_101 class from models module.")

        try:
            self.transform = transformator.getComposeTransformation()

        except AttributeError as _: # Transformator is a Compose
            self.transform = transformator

        self.num_workers = num_workers
        self.batch_size  = batch_size
        self.extensions  = extensions

        #if datasetModel is UCF_101:
        raw_path, dict_path = getPaths(preprocessed) #Raw path is preprocessed file informations.
        self.dataset = UCF_101(raw_path, dict_path, supportedExtensions= self.extensions, transform= self.transform)

    """
    Main method to get the Dataloader and to use it in the main method of train.
    """
    def getDataLoader(self):

        return DataLoader(self.dataset, batch_size= self.batch_size, num_workers= self.num_workers, shuffle= True, pin_memory= True, drop_last= True)