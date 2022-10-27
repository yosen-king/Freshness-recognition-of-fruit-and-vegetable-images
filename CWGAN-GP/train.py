import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import time
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
import os
import visdom
from models import *
from torch.autograd.variable import Variable
import cv2
import numpy as np

class Trainer():

    def __init__(self, sample_interval, data_directory, class_size, embedding_dim, batch_size, latent_size=100, device='cpu', lr=0.0002, num_workers=1):
        #load dataset
        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])

        #check if data directory exists, if not, create it.
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
            print('Directory created.')
        else:
            print('Directory exists.')

        #get dataset from directory. If not present, download to directory
        # self.dataset = torchvision.datasets.FashionMNIST(data_directory, train=True, transform=transformation, download=True)
        
        ## 引入自己的数据
        data_transforms = transforms.Compose([
                transforms.Resize((100,100)),  #280
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
        data_dir = 'archive'
        self.dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir),data_transforms)        

        # DataLoader(image_datasets_0, shuffle=True, batch_size=512)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        self.batch_size = batch_size
        self.device = device
        self.class_size = class_size
        self.sample_interval = sample_interval
        #define models
        self.latent_size = 100 

        self.dis = Discriminator(class_size, embedding_dim).to(device)
        self.gen = Generator(latent_size,class_size, embedding_dim).to(device)

        self.loss_func = nn.BCELoss().to(device)

        self.optimizer_d = optim.RMSprop(self.dis.parameters(), lr=lr)
        self.optimizer_g = optim.RMSprop(self.gen.parameters(), lr=lr)

    def gradient_penalty(self, real_samples, fake_samples, labels):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.dis(interpolates, labels)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        fake.requires_grad = False
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradients = gradients[0].view(gradients[0].size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    # 保存图片
    
    def save_fake_image(self,imgs_fake,labels,epoch,itera,idx_to_class_dict):
        for i in range(len(imgs_fake)):
            img_try = imgs_fake[i].cpu().detach().numpy()
            img_try = np.transpose(img_try,(1,2,0))
            name = '%d-%d-%d'%(epoch,itera,i)
            plt.imsave('fake_img/{}/{}.jpg'.format(idx_to_class_dict[labels[i].item()],name),
                       cv2.normalize(img_try, None, alpha=.001, beta=.999, norm_type=cv2.NORM_MINMAX),)
        
    def train(self, epochs, saved_image_directory, saved_model_directory):
        Tensor = torch.cuda.FloatTensor # if cuda else torch.FloatTensor
        try:
            import visdom
            vis = visdom.Visdom(server="http://localhost", port=8097, raise_exceptions=True) # Create vis env.
        except ImportError:
            vis = None
        else:
            vis.close(None) # Clear all figures.        
        
        loss_legend = ['Discriminator', 'Generator','Wasserstein']
        start_time = time.time()

        gen_loss_list = []
        dis_loss_list = []
        was_loss_list = []

        lmbda_gp = 10
        
        epochs_list = []
        
        idx_to_class_dict = {k:v for v,k in self.data_loader.dataset.class_to_idx.items()}
        for epoch in range(epochs):
            gen_loss = 0
            dis_loss = 0
            cur_time = time.time()
            numvis = 0  #用于visdom可视化
            for images, labels in self.data_loader:
                numvis +=1
                b_size = len(images)
                #train Discriminator with Wasserstein Loss
                self.optimizer_d.zero_grad()

                #fake loss
                z = torch.randn(b_size, self.latent_size).to(self.device)
                fake_images = self.gen(z, labels.to(self.device))
                fake_pred = self.dis(fake_images, labels.to(self.device))
                d_loss_fake = torch.mean(fake_pred)
                # print('ddddd')
                
                
                #real loss
                real_pred = self.dis(images.to(self.device), labels.to(self.device))
                d_loss_real = -torch.mean(real_pred)

                gp = self.gradient_penalty(images.to(self.device), fake_images, labels.to(self.device))

                d_loss = d_loss_fake - d_loss_real
                was_loss = (d_loss_fake + d_loss_real) + lmbda_gp*gp #剑减号？？？
                was_loss.backward()
                self.optimizer_d.step()

                dis_loss += d_loss.item()/b_size

               
                #train Generator
                self.optimizer_g.zero_grad()

                z = torch.randn(b_size, self.latent_size).to(self.device)
                fake_images = self.gen(z, labels.to(self.device))
                fake_pred = self.dis(fake_images, labels.to(self.device))
                g_loss = -torch.mean(fake_pred)
                g_loss.backward()
                self.optimizer_g.step()

                gen_loss += g_loss.item()/b_size

            cur_time = time.time() - cur_time
            print(len(self.data_loader))
            
            
            print('Epoch {},    Gen Loss: {:.4f},   Dis Loss: {:.4f},   Was Loss: {:.4f}'.format(epoch, gen_loss, dis_loss, was_loss))
            print('Time Taken: {:.4f} seconds. Estimated {:.4f} hours remaining'.format(cur_time, (epochs-epoch)*(cur_time)/3600))
            gen_loss_list.append(gen_loss)
            dis_loss_list.append(dis_loss)
            was_loss_list.append(was_loss)

            
            
            
            
            if vis:
                batches_done = epoch * len(self.data_loader) + numvis
                if batches_done % self.sample_interval == 0:  # self.sample_interval 这里还没解决！！！！！

                    labels = torch.LongTensor(np.arange(self.class_size)).to(self.device)  #这里要修改为分类数
                    z = torch.randn(self.class_size, self.latent_size).to(self.device)  #这里要修改为分类数
                    sample_images = self.gen(z, labels)


                    imgs_fake = Variable(sample_images)

                    # Keep a record of losses for plotting.
                    epochs_list.append(epoch + numvis/len(self.data_loader))
                    # g_losses.append(g_loss.item())
                    # d_losses.append(d_loss.item())

                    # Display results on visdom page.
                    print(dis_loss)
                    vis.line(
                        X=torch.stack([Tensor(epochs_list)] * len(loss_legend), dim=1),
                        Y=torch.stack((Tensor(dis_loss_list), Tensor(gen_loss_list), Tensor(was_loss_list)), dim=1),
                        
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
                        imgs_fake.data,
                        nrow=5, win=2,
                        opts={
                            'title': 'WGAN-GP output [Epoch {}]'.format(epoch),
                            'width': 512,
                            'height': 512,
                        }
                    ) 
            
            
            if epoch in [99,199,299,399,499]:



                #show samples
                labels = torch.LongTensor(np.arange(self.class_size)).to(self.device)  #这里要修改为分类数
                z = torch.randn(self.class_size, self.latent_size).to(self.device)  #这里要修改为分类数
                sample_images = self.gen(z, labels)

                imgs_fake = Variable(sample_images)
                self.save_fake_image(imgs_fake,labels,epoch,numvis,idx_to_class_dict)
                print('保存图像成功')

                #save models to model_directory
                torch.save(self.gen.state_dict(), saved_model_directory + '/generator_{}.pt'.format(epoch))
                torch.save(self.dis.state_dict(), saved_model_directory + '/discriminator_{}.pt'.format(epoch))

                image_grid = torchvision.utils.make_grid(sample_images.cpu().detach(), nrow=5, normalize=True)
                _, plot = plt.subplots(figsize=(12, 12))
                plt.axis('off')
                plot.imshow(image_grid.permute(1, 2, 0))
                plt.savefig(saved_image_directory + '/epoch_{}_checkpoint.jpg'.format(epoch), bbox_inches='tight')

        finish_time = time.time() - start_time
        print('Training Finished. Took {:.4f} seconds or {:.4f} hours to complete.'.format(finish_time, finish_time/3600))
        return gen_loss_list, dis_loss_list

def main():

    parser = argparse.ArgumentParser(description='Hyperparameters for training GAN')
    #hyperparameter loading
    parser.add_argument('--data_directory', type=str, default='data', help='directory to MNIST dataset files')
    parser.add_argument('--saved_image_directory', type=str, default='data/saved_images', help='directory to where image samples will be saved')
    parser.add_argument('--saved_model_directory', type=str, default='saved_models', help='directory to where model weights will be saved')
    parser.add_argument('--class_size', type=int, default=10, help='number of unique classes in dataset')
    parser.add_argument('--embedding_dim', type=int, default=10, help='size of embedding vector')
    parser.add_argument('--batch_size', type=int, default=64, help='size of batches passed through networks at each step')
    parser.add_argument('--latent_size', type=int, default=100, help='size of gaussian noise vector')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or gpu depending on availability and compatability')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate of models')
    parser.add_argument('--num_workers', type=int, default=0, help='workers simultaneously putting data into RAM')
    parser.add_argument('--epochs', type=int, default=100, help='number of iterations of dataset through network for training')
    parser.add_argument('--sample_interval', type=int, default=29, help='interval betwen image samples')  #改成batch数量
    
    parser.add_argument('--display_port', type=int, default=8097, help='where to run the visdom for visualization? useful if running multiple visdom tabs')
    parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
    args = parser.parse_args()

        
    data_dir = args.data_directory
    saved_image_dir = args.saved_image_directory
    saved_model_dir = args.saved_model_directory
    class_size = args.class_size
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    latent_size = args.latent_size
    device = args.device
    lr = args.lr
    num_workers = args.num_workers
    epochs = args.epochs
    sample_interval = args.sample_interval

    gan = Trainer(sample_interval,data_dir, class_size, embedding_dim, batch_size, latent_size, device, lr, num_workers)
    gen_loss_lost, dis_loss_list = gan.train(epochs, saved_image_dir, saved_model_dir)


if __name__ == "__main__":
    main()
