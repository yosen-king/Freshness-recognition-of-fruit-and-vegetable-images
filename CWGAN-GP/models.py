import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

'''Discriminator'''
class Discriminator(nn.Module):
    def __init__(self, class_size, embedding_dim):
        super(Discriminator, self).__init__()

        self.embedding = nn.Embedding(class_size, embedding_dim)

        
        self.seq = nn.Sequential(
            nn.Linear(30000 + embedding_dim, 15000),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(15000, 7500),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(7500, 3750),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(3750, 1875),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1875,900),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(900, 500),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(500, 250),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(250, 1),
            # nn.Sigmoid()
        )

    def forward(self, input, label):
        #embed label
        label = self.embedding(label)
        # print(label)
        #flatten image to 2d tensor
        input = input.view(input.size(0), -1)

        #concatenate image vector and label
        x = torch.cat([input, label], 1)
        result = self.seq(x)
        return result


'''Generator'''
class Generator(nn.Module):
    def __init__(self, latent_size, class_size, embedding_dim):
        super(Generator, self).__init__()

        self.embedding = nn.Embedding(class_size, embedding_dim)

        self.seq = nn.Sequential(
            nn.Linear(latent_size + embedding_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),

    
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(512, 1870),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1870, 3750),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(3750, 7500),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(7500, 15000),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(15000, 3*100*100),   #这里输出的是 n_features = opt.channels * opt.img_size * opt.img_size
            nn.Tanh()
        )

    def forward(self, input, label):
        #embed label
        label = self.embedding(label)
        # print(input.shape)

        #concatenate latent vector (input) and label
        x = torch.cat([input, label], 1)

        result = self.seq(x)
        result = result.view(-1, 3, 100, 100)  #这里输出的是 n_features = opt.channels * opt.img_size * opt.img_size
        return result