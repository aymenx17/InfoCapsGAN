import argparse
import os, random
import torch
import torchvision
import numpy as np
from torch.backends import cudnn
from torch import optim
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from PIL import Image
from network import *
from loss import CapsuleLoss
from dis_modules import squash
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import scipy.stats as stats


parser = argparse.ArgumentParser(description='Generative model based on Capsule and Mutual Information theories')

# model hyper-parameters
parser.add_argument('--image_size', type=int, default=28) # 28 for MNIST

# training hyper-parameters
parser.add_argument('--num_epochs', type=int, default=30) # 30 or 50 for MNIST
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--lrD', type=float, default=0.00002) # Learning Rate for D
parser.add_argument('--lrG', type=float, default=0.0002) # Learning Rate for G
parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam

# Generator and Discriminator hyperparameters
parser.add_argument('--dim_real', type=int, default=62)



# misc
parser.add_argument('--db', type=str, default='mnist')  # Model Tmp Save
parser.add_argument('--model_path', type=str, default='./models')  # Model Tmp Save
parser.add_argument('--sample_path', type=str, default='./results')  # Results
parser.add_argument('--sample_size', type=int, default=100)
parser.add_argument('--log_step', type=int, default=20)
parser.add_argument('--sample_step', type=int, default=100)


# deterministic DataLoader
def _init_fn(worker_id):
    torch.manual_seed(seed + worker_id)
    torch.cuda.manual_seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


##### Helper Function for GPU Training
def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

##### Helper Function for Math
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)
##### Helper Function for vectors distance

# InfoGAN Function (Multi-Nomial)
def gen_dc(n_size, dim):
    codes=[]
    code = np.zeros((n_size, dim))
    random_cate = np.random.randint(0, dim, n_size)
    code[range(n_size), random_cate] = 1.0
    codes.append(code)
    codes = np.concatenate(codes,1)
    return torch.Tensor(codes)




######################### Main Function
def main():
    # Pre-Settings
    cudnn.benchmark = True
    global args
    args = parser.parse_args()
    print(args)
    # seed for random generator libraries
    global seed
    #seed = np.random.randint(0, 10000)
    seed = 9345
    print(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # full numpy array Print
    np.set_printoptions(threshold=np.inf)

    # deterministic cudnn
    print('Additional cudnn determinism')
    torch.backends.cudnn.deterministic = True


    transform = transforms.Compose([
        transforms.Scale((args.image_size, args.image_size)),
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ])
    

    #tensorboardX
    writer = SummaryWriter()

    dataset = datasets.MNIST('./', train=True, transform=transform, target_transform=None, download=True)


    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)

    # Networks
    generator = Generator(in_caps=6*6*32, num_caps=10, in_dim=8, dim_caps=16, dim_real=args.dim_real)
    #generator_cnn = GeneratorCNN(z_dim=32)
    discriminator = Discriminator(img_shape=(1,28,28), channels=256, primary_dim=8, num_classes=10, out_dim=16, dim_real=args.dim_real, num_routing=3)

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), args.lrG, [args.beta1, args.beta2])
    #gcnn_optimizer = optim.Adam(generator_cnn.parameters(), args.lrG, [args.beta1, args.beta2])
    d_optimizer = optim.Adam(discriminator.parameters(), args.lrD, [args.beta1, args.beta2])

    if torch.cuda.is_available():
        generator.cuda()
        #generator_cnn.cuda()
        discriminator.cuda()

    # setup loss function
    criterion = nn.BCELoss().cuda()
    margin = CapsuleLoss()


    total_step = len(data_loader)  # For Print Log
    for epoch in range(args.num_epochs):
        for i, images in enumerate(data_loader):
            # ===================== Train D =====================#
            if args.db == 'mnist': # To Remove Label
                images = to_variable(images[0])
            else:
                images = to_variable(images)

            batch_size = images.size(0)


            # reality_structure
            real_struc = torch.randn(batch_size, 1, args.dim_real - 10) # -> (batch_size, 1, dim_real)
            real_struc = to_variable(real_struc)


            dc = gen_dc(batch_size, 10)
            dc = to_variable(dc)

            real_struc = torch.cat((real_struc, dc.unsqueeze(1)), dim=-1)
            real_struc = squash(real_struc)

            #
            fake_images, gp_caps, gc_caps = generator(real_struc, epoch)
            d_out_norm_real, dp_out_real, dc_out_real, dr_out_real, sf_real = discriminator(images)
            d_out_norm_fake, dp_out_fake, dc_out_fake, dr_out_fake, sf_fake = discriminator(fake_images)


            # Mutual Information Loss
            #d_loss_dc = -(torch.mean(torch.sum(dc * sf_fake, 1)) + 1)
            d_loss_dc = margin(sf_fake, dc)*1e-02 + 1


            # Euclidean distance

            # classes capsules
            gc_caps= gc_caps.squeeze(1)
            dist_c_caps = torch.sum((dc_out_fake - gc_caps)**2, -1)**0.5
            loss_c_caps = torch.mean(torch.sum(dist_c_caps, -1))

            # primary capsules
            dist_p_caps = torch.sum((gp_caps - dp_out_fake)**2, -1)**0.5
            loss_p_caps = torch.mean(torch.sum(dist_p_caps, -1))

            # Cosine Similarity
            # classes capsules
            cos_c = F.cosine_similarity(gc_caps, dc_out_fake, dim=-1)
            cos_c = torch.mean(torch.sum(cos_c, dim=-1))
            # primary capsules
            cos_p = F.cosine_similarity(gp_caps, dp_out_fake, dim=-1)
            cos_p = torch.mean(torch.sum(cos_p, dim=-1))




            d_loss_a = -torch.mean(torch.log(d_out_norm_real[:,0]) + torch.log(1 - d_out_norm_fake[:,0]))


            d_loss = d_loss_a + 1.0*d_loss_dc



            # Optimization
            discriminator.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

            # ===================== Train G =====================#
            # Fake -> Real
            g_loss_a = -torch.mean(torch.log(d_out_norm_fake[:,0]))

            g_loss = g_loss_a + 1.0*d_loss_dc


            # Optimization
            generator.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Tensorboard scalars
            writer.add_scalars('Losses', {'Discriminator': d_loss, 'Generator': g_loss},i+1)

            if (i + 1) % (args.log_step + 100) == 0:
                # print c caps of discriminator
                l1, l2, l3, l4 = [], [], [], []
                c_norm_real = torch.norm(dc_out_real, dim=-1)
                c_norm_fake = torch.norm(dc_out_fake, dim=-1)
                cg_norm = torch.norm(gc_caps, dim=-1)
                for k in range(c_norm_fake.size()[1]):
                    l1.append(round(c_norm_real[0, k].item(), 5))
                    l2.append(round(c_norm_fake[0, k].item(), 5))
                    l3.append(round(cg_norm[0, k].item(), 5))
                    l4.append(dc[0,k].item())

                print('\n\nc_norm_real {} \n\nc_norm_fake {} \n\ncg_norm {} \n\ndc {}'.format(l1,l2,l3,l4))
                print('*'*100)
                _, freq_cls = torch.max(sf_real, dim=-1)
                freq_cls = stats.itemfreq(freq_cls.cpu())
                print(freq_cls)
                print('*'*100)

            if (i + 1) % (args.log_step + 400) == 0:
                l1, l2 = [], []
                randint = [np.random.randint(0, gp_caps.size()[1]) for i in range(300)]
                p_norm_fake = torch.norm(dp_out_fake, dim=-1)
                gp_norm = torch.norm(gp_caps, dim=-1)
                for k in range(300):
                    l1.append(round(p_norm_fake[0, randint[k]].item(), 5))
                    l2.append(round(gp_norm[0, randint[k]].item(), 5))
                print('*'*100)
                print('p_norm_fake\n {}'.format(l1))
                print('-'*100)
                print('gp_norm\n {}'.format(l2))
                print('*'*100)



            # print the log info
            if (i + 1) % args.log_step == 0:
                print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, dc_loss: %.4f, loss_p_caps: %.4f, loss_c_caps: %.4f, cos_p: %.4f, cos_c: %.4f'
                      % (epoch + 1, args.num_epochs, i + 1, total_step, d_loss, g_loss, d_loss_dc, loss_p_caps, loss_c_caps, cos_p, cos_c ))

            # save the sampled images (10 Category(Discrete), 10 Continuous Code Generation : 10x10 Image Grid)
            if (i + 1) % args.sample_step == 0:

                real_struc = torch.randn(100, 1, args.dim_real - 10)
                tmp = np.zeros((100, 10))
                for k in range(10):
                    tmp[k * 10:(k + 1) * 10, k] = 1
                tmp = torch.Tensor(tmp)
                real_struc = torch.cat((real_struc, tmp.unsqueeze(1)), dim=-1)
                real_struc = to_variable(real_struc)
                real_struc = squash(real_struc)

                fake_images, _, _ = generator(real_struc, epoch)
                torchvision.utils.save_image(denorm(fake_images.data),
                                             os.path.join(args.sample_path,
                                                          'generated-%d-%d.png' % (epoch + 1, i + 1)), nrow=10)

        # save the model parameters for each epoch
        g_path = os.path.join(args.model_path, 'generator-%d.pkl' % (epoch + 1))
        torch.save(generator.state_dict(), g_path)




if __name__ == "__main__":
    main()
    writer.close()
