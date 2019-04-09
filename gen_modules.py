import torch
import torch.nn as nn
import torch.nn.functional as F
from network import *
from main import squash


class GenCapsules(nn.Module):

	def __init__(self, in_caps, num_caps, in_dim, dim_caps, dim_real):
		"""
		Initialize the layer.

		Args:
		in_dim: 		8
		in_caps: 		6*6*32 --> 1152
		num_caps: 		10
		dim_caps: 		16

		"""
		super(GenCapsules, self).__init__()

		self.dim_real = dim_real
		self.W1 = nn.Parameter(torch.randn(1, 6*6*32, num_caps, in_dim, dim_caps)*(3/(in_dim + dim_caps + 6*6*32))**0.5)
		self.W0 = nn.Parameter(torch.randn(1, num_caps, 1, dim_caps, dim_real)*(3/(dim_caps + num_caps + dim_real))**0.5)


		self.dconv1 = nn.ConvTranspose2d(256, 1, 9, 1, 0)
		self.dconv0 = nn.ConvTranspose2d(256, 256, 10, 2, 0)
		torch.nn.init.xavier_normal_(self.dconv1.weight)
		torch.nn.init.xavier_normal_(self.dconv0.weight)
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()

		self.batchnorm = nn.BatchNorm2d(256)
		self.batchnorm0 = nn.BatchNorm2d(256)

	def forward(self, x, epoch):

		batch_size = x.size()[0]

		# classes capsules

		# W0 @ real_struc =
		# (1, num_caps, in_caps, dim_caps, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
		# (batch_size, num_caps, in_caps, dim_caps, 1)
		x = x.unsqueeze(1).unsqueeze(-1)
		c_caps = torch.matmul(self.W0, x)

		# (batch_size, num_caps, in_caps, dim_caps)
		c_caps = c_caps.transpose(1,2).transpose(-2,-1)




		# squash
		# (batch_size, 1, 10, 16, 1)
		c_caps = squash(c_caps).transpose(-2,-1)


		# primary capsules

		# W1 @ c_caps =
		# (1, in_caps, num_caps, in_dim, dim_caps) @ (batch_size, 1, in_caps, in_dim, 1) =
		# (batch_size, in_caps, num_caps, in_dim, 1)
		p_caps = torch.matmul(self.W1, c_caps)
		c_caps = c_caps.squeeze(1).squeeze(-1)

		# (batch_size, in_caps, num_caps, in_dim)
		p_caps = p_caps.squeeze(-1)


		# sum projected vectors
		p_caps = p_caps.sum(dim=2)

		# squash
		p_caps = squash(p_caps)


		# reshape capsules for convolutional operations
		# (batch_size, in_caps, in_dim) -> (batch_size, 32, 6, 6, 8) -> (batch_size, 256, 6, 6)
		out = p_caps.view(p_caps.size(0), 32, 6, 6, 8)
		out = out.view(p_caps.size(0), 256, 6, 6)


		# apply deconvs
		out = self.dconv0(out)
		out = self.batchnorm(out)
		out = self.relu(out)

		out = self.dconv1(out)
		out = self.tanh(out)

		return out, p_caps, c_caps
