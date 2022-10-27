'''
Although I tried to implement this exactly as it was described in the paper,
I couldn't get over the many problems such as mode collapse, diverging generator etc with the described network.
I wasn't able to stabilize the training with SGD. I don't understand why a
maxout layer (already nonlinear) has to be followed by a relu layer. I also had to use batchnorm for stability (instead of dropout).
This paper brings the simple (yet effective, especially if you implemented on more
complex networks such as pix2pix or any visual conditional generation system)
idea of concatenating condition (e.g., labels) to the input to get conditional outputs,
yet I think has fundamental flaws in designing the architecture.
'''
import argparse

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: beta 1')
parser.add_argument('--b2', type=float, default=0.999, help='adam: beta 2')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=200, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--n_classes', type=int, default=12, help='number of classes (e.g., digits 0 ..9, 10 classes on mnist)')
parser.add_argument('--display_port', type=int, default=8097, help='where to run the visdom for visualization? useful if running multiple visdom tabs')
parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
parser.add_argument('--sample_interval', type=int, default=29, help='interval betwen image samples')  #改成batch数量
opt = parser.parse_known_args()[0]


try:
    import visdom
    vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, raise_exceptions=True) # Create vis env.
except ImportError:
    vis = None
else:
    vis.close(None) # Clear all figures.


img_dims = (opt.channels, opt.img_size, opt.img_size)
n_features = opt.channels * opt.img_size * opt.img_size


'''
保存图每一场生成图像
'''

import matplotlib.pyplot as plt
import numpy as np
import cv2

def save_fake_image(imgs_fake,labels,epoch,itera):
    for i in range(len(imgs_fake)):
        img_try = imgs_fake[i].cpu().detach().numpy()
        img_try = np.transpose(img_try,(1,2,0))
        name = '%d-%d-%d'%(epoch,itera,i)
        plt.imsave('fake_img/{}/{}.jpg'.format(idx_to_class_dict[labels[i].item()],name),
                   cv2.normalize(img_try, None, alpha=.001, beta=.999, norm_type=cv2.NORM_MINMAX),)
        
def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Map z & y (noise and label) into the hidden layer.
        # TO DO: How to run this with a function defined here?
        self.z_map = nn.Sequential(
            nn.Linear(opt.latent_dim, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
        )
        self.y_map = nn.Sequential(
            nn.Linear(opt.n_classes, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
        )
        self.zy_map = nn.Sequential(
            nn.Linear(1200, 1200),
            nn.BatchNorm1d(1200),
            nn.ReLU(inplace=True),
        )

        self.model = nn.Sequential(
            nn.Linear(1200, n_features),
            nn.Tanh()
        )
        # Tanh > Image values are between [-1, 1]


    def forward(self, z, y):
        zh = self.z_map(z)
        yh = self.y_map(y)
        zy = torch.cat((zh, yh), dim=1) # Combine noise and labels.
        zyh = self.zy_map(zy)
        x = self.model(zyh)
        x = x.view(x.size(0), *img_dims)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(240, 1),
            nn.Sigmoid()
        )

        # Imitating a 3d array by combining second and third dimensions via multiplication for maxout.
        self.x_map = nn.Sequential(nn.Linear(n_features, 240 * 5))
        self.y_map = nn.Sequential(nn.Linear(opt.n_classes, 50 * 5))
        self.j_map = nn.Sequential(nn.Linear(240 + 50, 240 * 4))

    def forward(self, x, y):
        # maxout for x
        x = x.view(-1, n_features)
        x = self.x_map(x)
        x, _ = x.view(-1, 240, 5).max(dim=2) # pytorch outputs max values and indices
        # .. and y
        y = y.view(-1, opt.n_classes)
        y = self.y_map(y)
        y, _ = y.view(-1, 50, 5).max(dim=2)
        # joint maxout layer
        jmx = torch.cat((x, y), dim=1)
        jmx = self.j_map(jmx)
        jmx, _ = jmx.view(-1, 240, 4).max(dim=2)

        prob = self.model(jmx)
        return prob


def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
#          transforms.Lambda(lambda x: x.repeat(3,1,1)),
#          transforms.Normalize((0.1307,), (0.3081,))
         transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
         ])
    output_dir = './data/mnist'
    return datasets.MNIST(root=output_dir, train=True,
                          transform=compose, download=True)

# mnist = mnist_data()


data_transforms = transforms.Compose([
        transforms.Resize((200,200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

# batch_size=32
data_dir = 'archive'
image_datasets_0 = datasets.ImageFolder(os.path.join(data_dir),data_transforms)
batch_iterator = DataLoader(image_datasets_0, shuffle=True, batch_size=512) #  opt.batch_size List, NCHW format.
idx_to_class_dict = {k:v for v,k in image_datasets_0.class_to_idx.items()}



image_datasets_0.class_to_idx



batch_iterator





cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
gan_loss = nn.BCELoss()

generator = Generator()
discriminator = Discriminator()

optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


# Loss record.
g_losses = []
d_losses = []
epochs = []
loss_legend = ['Discriminator', 'Generator']

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()

# Weight initialization.
generator.apply(weights_init_xavier)
discriminator.apply(weights_init_xavier)


for epoch in range(opt.n_epochs):
    print('Epoch {}'.format(epoch))
    for i, (batch, labels) in enumerate(batch_iterator):

        real = Variable(Tensor(batch.size(0), 1).fill_(1), requires_grad=False)
        fake = Variable(Tensor(batch.size(0), 1).fill_(0), requires_grad=False)

        labels_onehot = Variable(Tensor(batch.size(0), opt.n_classes).zero_())
        labels_ = labels.type(LongTensor) # Note: MNIST dataset is given as a longtensor, and we use .type() to feed it into cuda (if cuda available).
        labels_ = labels_.view(batch.size(0), 1)
        labels_onehot = labels_onehot.scatter_(1, labels_, 1) # As of 0.4, we don't have one-hot built-in function yet in pytorch.

        imgs_real = Variable(batch.type(Tensor))
        noise = Variable(Tensor(batch.size(0), opt.latent_dim).normal_(0, 1))
        imgs_fake = generator(noise, labels_onehot)

        # == Discriminator update == #
        optimizer_D.zero_grad()

        # A small reminder: given classes c, prob. p, - c*logp - (1-c)*log(1-p) is the BCE/GAN loss.
        # Putting D(x) as p, x=real's class as 1, (..and same for fake with c=0) we arrive to BCE loss.
        # This is intuitively how well the discriminator can distinguish btw real & fake.
        d_loss = gan_loss(discriminator(imgs_real, labels_onehot), real) + \
                 gan_loss(discriminator(imgs_fake, labels_onehot), fake)

        d_loss.backward()
        optimizer_D.step()


        # == Generator update == #
        noise = Variable(Tensor(batch.size(0), opt.latent_dim).normal_(0, 1))
        imgs_fake = generator(noise, labels_onehot)

        optimizer_G.zero_grad()

        # Minimizing (1-log(d(g(noise))) is less stable than maximizing log(d(g)) [*].
        # Since BCE loss is defined as a negative sum, maximizing [*] is == minimizing [*]'s negative.
        # Intuitively, how well does the G fool the D?
        g_loss = gan_loss(discriminator(imgs_fake, labels_onehot), real)

        g_loss.backward()
        optimizer_G.step()

        if vis:
            batches_done = epoch * len(batch_iterator) + i
            if batches_done % opt.sample_interval == 0:

                # Generate digits from 0, ... 9, 5 samples in each column.
                noise = Variable(Tensor(5*12, opt.latent_dim).normal_(0, 1))

                labels_onehot = Variable(Tensor(5*12, opt.n_classes).zero_())
                labels_ = torch.range(0, 11)
                labels_ = labels_.view(1, -1).repeat(5, 1).transpose(0, 1).contiguous().view(1, -1) # my version of matlab's repelem
                labels_ = labels_.type(LongTensor)
                labels_ = labels_.view(-1, 1)
                labels_onehot = labels_onehot.scatter_(1, labels_, 1) # As of 0.4, we don't have one-hot built-in function yet in pytorch.

                imgs_fake = Variable(generator(noise, labels_onehot))

                # Keep a record of losses for plotting.
                epochs.append(epoch + i/len(batch_iterator))
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

                # Display results on visdom page.
                vis.line(
                    X=torch.stack([Tensor(epochs)] * len(loss_legend), dim=1),
                    Y=torch.stack((Tensor(d_losses), Tensor(g_losses)), dim=1),
                    opts={
                        'title': 'loss over time',
                        'legend': loss_legend,
                        'xlabel': 'epoch',
                        'ylabel': 'loss',
                        'width': 512,
                        'height': 512
                    },
                    win=1)
                vis.images(
                    imgs_fake.data[:60],
                    nrow=5, win=2,
                    opts={
                        'title': 'GAN output [Epoch {}]'.format(epoch),
                        'width': 512,
                        'height': 512,
                    }
                )
                # vis.images(
                #     imgs_real.data[:50],
                #     nrow=5, win=3,
                #     opts={
                #         'title': 'input [Epoch {}]'.format(epoch),
                #         'width': 512,
                #         'height': 512,
                #     }
                # )
         # save fake image（not from random noise, but original image data from bratch）
        if epoch == 499:
            save_fake_image(imgs_fake,labels,epoch,i)
            print('保存图像成功')
            
    # save model
    if epoch in [99,199,299,399,499]:
        torch.save(generator, "models_weight/CGANgenerator_%d.pth" % epoch)
        print('保存model成功')
        
        
        