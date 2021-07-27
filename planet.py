import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from typing import Any, List, Tuple, Optional

class Reshape(nn.Module):

    def __init__(self, shape: List):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Encoder(nn.Module):
    """ As mentioned in the paper, VAE is used
        to calculate the state posterior needed 
        for parameter learning in RSSM. The training objective
        here is to create bound on the data. Here losses
        are written only for observations as rewards losses
        follow them. """

    def __init__(self, 
                    depth : int = 32, 
                    input_channels : Optional[int] = 3):
        super(Encoder, self).__init__()
        """
        Initialize the parameters of the Encoder.
        Args
            depth (int) : Number of channels in the first convolution layer.
            input_channels (int) : Number of channels in the input observation.
        """
        self.depth = depth
        self.input_channels = input_channels
        self.encoder = nn.Sequential(
                nn.Conv2d(self.input_channels, self.depth, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(self.depth, self.depth * 2, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(self.depth * 2, self.depth * 4, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(self.depth * 4, self.depth * 8, 4, stride=2),
                nn.ReLU(),
        )
    
    def forward(self, x):
        """ Flatten the input observation [batch, horizon, 3, 64, 64]
            into shape [batch * horizon, 3, 64, 64] before feeding it
            to the input. """
        orig_shape = x.shape
        x = x.reshape(-1, *x.shape[-3:])
        x = self.encoder(x)
        x = x.reshape(*orig_shape[:-3], -1)
        return x


class Decoder(nn.Module):
    """
    Takes the input from the RSSM model
    and then decodes it back to images from
    the latent space model. It is mainly used
    in calculating losses.
    """
    def __init__(self,
                    input_size : int,
                    depth: int = 32,
                    shape: Tuple[int] = (3, 64, 64)):
        super(Decoder, self).__init__()
        self.depth = depth
        self.shape = shape
        self.decoder = nn.Sequential(
            nn.Linear(input_size, 32 * self.depth),
            nn.Reshape([-1, 32 * self.depth, 1, 1]),
            nn.ConvTranspose2d(32 * self.depth, 4 * self.depth, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * self.depth, 2 * self.depth, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * self.depth, self.depth, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.depth, self.shape[0], 6, stride=2),
        )
    
    def forward(self, x):
        orig_shape = x.shape
        x = self.model(x)
        reshape_size = orig_shape[:-1] + self.shape
        mean = x.view(*reshape_size)
        return td.Independent(  
                        td.normal(mean, 1), 
                        len(self.shape))


class ActionDecoder(nn.Module):
    """
    """
    def __init__(self,):
        super(ActionDecoder, self).__init__()
    
    def forward(self, x):
        return x


class RewardDecoder(nn.Module):
    """
    """
    def __init__(self,):
        super(RewardDecoder, self).__init__()
    
    def forward(self, x):
        return x


class ValueDecoder(nn.Module):
    """
    """
    def __init__(self,):
        super(ValueDecoder, self).__init__()
    
    def forward(self, x):
        return x


class RSSM(nn.Module):
    """
    """
    def __init__(self,):
        super(RSSM, self).__init__()
    
    def forward(self, x):
        return x


class PLANet(nn.Module):
    
    def __init__(self,):
        super(PLANet, self).__init__()
    
    def forward(self, x):
        return x