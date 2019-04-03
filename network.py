import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
import dis_modules as Dis
import gen_modules as Gen
from dis_modules import squash
from torch.nn.init import kaiming_normal, calculate_gain


class Discriminator(nn.Module):
	def __init__(self, img_shape, channels, primary_dim, num_classes, out_dim, dim_real, num_routing, kernel_size=9):
		super(Discriminator, self).__init__()
		self.img_shape = img_shape
		self.num_classes = num_classes
		self.conv1 = nn.Conv2d(img_shape[0], channels, kernel_size, stride=1, bias=True)
		torch.nn.init.xavier_normal_(self.conv1.weight)
		self.relu = nn.ReLU(inplace=True)
		self.dim_real = dim_real
		self.primary = Dis.PrimaryCapsules(channels, channels, primary_dim, kernel_size)
		self.batchnorm = nn.BatchNorm2d(channels)

		primary_caps = int(channels / primary_dim * ( img_shape[1] - 2*(kernel_size-1) ) * ( img_shape[2] - 2*(kernel_size-1) ) / 4)
		self.digits = Dis.RoutingCapsules(primary_dim, primary_caps, num_classes, out_dim, num_routing)
		self.real = Dis.RealOrFake(num_classes, out_dim, self.dim_real, num_routing)
		self.convR = nn.Conv1d(52, 1, 1)
	def forward(self, x):
		out = self.conv1(x)
		out = self.batchnorm(out)
		out = self.relu(out)
		p_caps = self.primary(out)
		c_caps = self.digits(p_caps)
		c_caps = squash(c_caps)
		norm_c = torch.norm(c_caps, dim=-1)
		c_caps = (norm_c.unsqueeze(-1) + 1)*c_caps
		r_caps = self.real(c_caps) # -> (batch_size, 1, dim_real)

		preds = self.convR(r_caps.transpose(1,2)).squeeze(-1)
		preds = torch.sigmoid(preds)


		return preds, p_caps, c_caps, r_caps, norm_c


class Generator (nn.Module):
	def __init__(self, in_caps, num_caps, in_dim, dim_caps, dim_real):
		super(Generator, self).__init__()
		self.in_caps = in_caps
		self.num_caps = num_caps
		self.in_dim = in_dim
		self.dim_caps = dim_caps
		self.dim_real = dim_real

		self.gen = Gen.GenCapsules(in_caps, num_caps, in_dim, dim_caps, dim_real)

	def forward (self, x, epoch):

		# explore and generate data
		out, p_caps, c_caps = self.gen(x, epoch)

		return out, p_caps, c_caps
