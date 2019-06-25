from mocogan.models import Discriminator_V, Discriminator_I, Generator_I, GRU

import torch.nn as nn, optim

class Trainer(nn.Module):

    def __init__(self, discriminator_i, discriminator_v, generator_i, gru, 
                 optimParameters = {
                     "dis_i": { "optim": optim.Adam, "lr": 0.0002, "betas": (0.5, 0.999), "weight_decay": 0.00001 }, 
                     "dis_v": { "optim": optim.Adam, "lr": 0.0002, "betas": (0.5, 0.999), "weight_decay": 0.00001 },
                     "gen_i": { "optim": optim.Adam, "lr": 0.0002, "betas": (0.5, 0.999), "weight_decay": 0.00001 }
                     "gru": { "optim": optim.Adam, "lr": 0.0002, "betas": (0.5, 0.999), "weight_decay": 0.00001 }
                     }
                 ):
                     
        super(Trainer, self).__init__()
        self.discriminator_i = discriminator_i
        self.discriminator_v = discriminator_v
        self.generator_i     = generator_i
        self.gru             = gru
        
        for key, internalDictionary in optimParameters.items():
            
            optimizer = internalDictionary["optim"](self.discriminator_i.parameters(), lr = internalDictionary["lr"], betas = internalDictionary["betas"], weight_decay = internalDictionary["weight_decay"])
            
            self.optim_discriminator_i = optimizer if key == "dis_i" else self.optim_discriminator_i
            self.optim_discriminator_v = optimizer if key == "dis_v" else self.optim_discriminator_v
            self.optim_generator_i     = optimizer if key == "gen_i" else self.optim_generator_i
            self.optim_gru             = optimizer if key == "gru"   else self.optim_gru
        
        
    def train_discriminator(self):
        
        
        
        
    
