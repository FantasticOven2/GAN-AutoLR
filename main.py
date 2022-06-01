import argparse
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image

from networks import Generator, Discriminator
from utils import get_data_loader, generate_images, save_gif
from auto_lrschedule.BO import BayesianOptimizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from skopt import gp_minimize
import copy

from loss import MCRGANloss

ADJ_LR = 200

mcr_loss = MCRGANloss(numclasses=2)

def explore_this_lr_G(netD, netG, lr, valloader, device='cuda'):
        optimizerD = optim.Adam(netD.parameters(), lr=lr[0])
        optimizerG = optim.Adam(netG.parameters(), lr=lr[0])
        real_label, fake_label = 1, 0
        netD_copy = copy.deepcopy(netD)
        # for _ in range(exploration_epoch):
        init_lossG, end_lossG = 0, 0
        for i, (real_images, _) in enumerate(valloader):
            bs = real_images.shape[0]
            netD.zero_grad()
            real_images = real_images.to(device)
            label = torch.full((bs,), real_label, dtype=torch.float, device=device)

            output = netD(real_images)
            # print('real output: ', output)
            lossD_real = criterion(output, label)
            # lossD_real = mcr_loss(output_real, output_real, label, i, 1)
            lossD_real.backward()

            noise = torch.randn(bs, opt.nz, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(fake_label)
            output = netD(fake_images.detach())
            # print('fake output: ', output)
            lossD_fake = criterion(output, label)
            # lossD_fake = mcr_loss(output_real, output_fake, label, i, 1)
            lossD_fake.backward()
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake_images)
            lossG = criterion(output, label)
            # lossG = mcr_loss(output_real, output_fake, label, i, 1)
            lossG.backward()
            optimizerG.step()
        # print(lossD.cpu().detach().item())
        output = netD_copy(fake_images)
        lossG_copy = criterion(output, label)
        # print('lr: ', lr)
        lossG = lossG.cpu().detach().item()
        # print('lossG before: ', lossG)
        # print('lossD after: ', lossD + 1e-4)
        # return 5 * lossG - lossG_copy.cpu().detach().item()
        print('loss G: ', lossG)
        print('loss G cp: ', lossG_copy.cpu().detach().item())
        return -5 * lossG

# def explore_this_lr_D(netD, netG, lr, valloader, device='cuda', ld):
#         optimizerD = optim.Adam(netD.parameters(), lr=lr[0])
#         optimizerG = optim.Adam(netG.parameters(), lr=lr[0])
#         real_label, fake_label = 1, 0
#         # for _ in range(exploration_epoch):
#         for i, (real_images, _) in enumerate(valloader):
#             bs = real_images.shape[0]
#             netD.zero_grad()
#             real_images = real_images.to(device)
#             label = torch.full((bs,), real_label, dtype=torch.float, device=device)

#             output = netD(real_images)
#             print('real output: ', output)
#             lossD_real = criterion(output, label)
#             # lossD_real = mcr_loss(output_real, output_real, label, i, 1)
#             lossD_real.backward()

#             noise = torch.randn(bs, opt.nz, 1, 1, device=device)
#             fake_images = netG(noise)
#             label.fill_(fake_label)
#             output = netD(fake_images.detach())
#             print('fake output: ', output)
#             lossD_fake = criterion(output, label)
#             # lossD_fake = mcr_loss(output_real, output_fake, label, i, 1)
#             lossD_fake.backward()
#             lossD = lossD_real + lossD_fake
#             optimizerD.step()

#             netG.zero_grad()
#             label.fill_(real_label)
#             output = netD(fake_images)
#             lossG = criterion(output, label)
#             # lossG = mcr_loss(output_real, output_fake, label, i, 1)
#             lossG.backward()
#             optimizerG.step()
#         lossD = lossD.cpu().detach().item() + 1e-4
#         return ld - lossD

def search_lr(netD, netG, current_lr, val_loader):
        # n_iter = search_iter

        lr_lb = current_lr/10
        lr_hb = current_lr*10

        # kernel = C(1.0, (1e-5, 1e5)) * RBF(10, (1e-2, 1e2))
        # gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=10)

        # bo = BayesianOptimizer(lambda x: explore_this_lr(netD, netG, x, data, label),
        #                         gp,
        #                         mode="linear",
        #                         bound=[lr_lb, lr_hb])

        # found_lr, _  = bo.eval(n_iter=10)

        found_lr_G = gp_minimize(lambda x: explore_this_lr_G(copy.deepcopy(netD), copy.deepcopy(netG), x, val_loader),                  # the function to minimize
                  [(lr_lb, lr_hb)],      # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=15,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=1234)
        # print('found_lr G: ', found_lr_G.x[0])

        # found_lr_D = gp_minimize(lambda x: explore_this_lr_D(copy.deepcopy(netD), copy.deepcopy(netG), x, val_loader),                  # the function to minimize
        #           [(lr_lb, lr_hb)],      # the bounds on each dimension of x
        #           acq_func="EI",      # the acquisition function
        #           n_calls=15,         # the number of evaluations of f
        #           n_random_starts=5,  # the number of random initialization points
        #           noise=0.1**2,       # the noise level (optional)
        #           random_state=1234)
        # print('found_lr D: ', found_lr_D.x[0])        
        return found_lr_G.x[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCGANS MNIST')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--ndf', type=int, default=32, help='Number of features to be used in Discriminator network')
    parser.add_argument('--ngf', type=int, default=32, help='Number of features to be used in Generator network')
    parser.add_argument('--nz', type=int, default=100, help='Size of the noise')
    parser.add_argument('--d-lr', type=float, default=0.0002, help='Learning rate for the discriminator')
    parser.add_argument('--g-lr', type=float, default=0.0002, help='Learning rate for the generator')
    parser.add_argument('--nc', type=int, default=1, help='Number of input channels. Ex: for grayscale images: 1 and RGB images: 3 ')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--num-test-samples', type=int, default=16, help='Number of samples to visualize')
    parser.add_argument('--output-path', type=str, default='./results/', help='Path to save the images')
    parser.add_argument('--fps', type=int, default=5, help='frames-per-second value for the gif')
    parser.add_argument('--use-fixed', action='store_true', help='Boolean to use fixed noise or not')

    opt = parser.parse_args()
    print(opt)

    # Gather MNIST Dataset    
    train_loader, test_loader, val_loader = get_data_loader(opt.batch_size)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)

    # Define Discriminator and Generator architectures
    netG = Generator(opt.nc, opt.nz, opt.ngf).to(device)
    netD = Discriminator(opt.nc, opt.ndf).to(device)

    # loss function
    criterion = nn.BCELoss()

    # optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=opt.d_lr)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.g_lr)
    
    # initialize other variables
    real_label = 1
    fake_label = 0
    num_batches = len(train_loader)
    fixed_noise = torch.randn(opt.num_test_samples, 100, 1, 1, device=device)
    global_step = 0

    for epoch in range(opt.num_epochs):

        for i, (real_images, _) in enumerate(train_loader):
            bs = real_images.shape[0]
            real_images = real_images.to(device)

            if global_step % ADJ_LR == 0 and global_step != 0:
                lr_old = optimizerD.param_groups[0]['lr']
                lr_G = search_lr(copy.deepcopy(netD), copy.deepcopy(netG), lr_old, val_loader)
                print('G learning rate has been updated from: ', lr_old, 'to', lr_G)
                # print('D learning rate has been updated from: ', lr_old, 'to', lr_D)
                for g in optimizerD.param_groups:
                    g['lr'] = lr_G
                for g in optimizerG.param_groups:
                    g['lr'] = lr_G
            
            global_step += 1
            
            ##############################
            #   Training discriminator   #
            ##############################

            netD.zero_grad()
            label = torch.full((bs,), real_label, dtype=torch.float, device=device)
            output = netD(real_images)
            # print('real output: ', output)
            lossD_real = criterion(output, label)
            # lossD_real = mcr_loss(output_real, output_real, label, i, 1)
            lossD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(bs, opt.nz, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(fake_label)
            output = netD(fake_images.detach())
            # print('fake output: ', output)
            lossD_fake = criterion(output, label)
            # lossD_fake = mcr_loss(output_real, output_fake, label, i, 1)
            lossD_fake.backward()
            D_G_z1 = output.mean().item()
            lossD = lossD_real + lossD_fake
            optimizerD.step()
            
            ##########################
            #   Training generator   #
            ##########################

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake_images)
            lossG = criterion(output, label)
            # lossG = mcr_loss(output_real, output_fake, label, i, 1)
            lossG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if (i+1)%100 == 0:
                print('Epoch [{}/{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, Discriminator - D(G(x)): {:.2f}, Generator - D(G(x)): {:.2f}'.format(epoch+1, opt.num_epochs, 
                                                            i+1, num_batches, lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))
        netG.eval()
        generate_images(epoch, opt.output_path, fixed_noise, opt.num_test_samples, netG, device, use_fixed=opt.use_fixed)
        netG.train()

    # Save gif:
    save_gif(opt.output_path, opt.fps, fixed_noise=opt.use_fixed)