import torch
import torch.nn as nn
import torch.nn.functional as F
from network import *

def squash(s, dim=-1):
	'''
	"Squashing" non-linearity that shrunks short vectors to almost zero length and long vectors to a length slightly below 1
	Eq. (1): v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||

	Args:
	s: 	Vector before activation
	dim:	Dimension along which to calculate the norm

	Returns:
	Squashed vector
	'''
	squared_norm = torch.sum(s**2, dim=dim, keepdim=True)
	return squared_norm / (1 + squared_norm) * s / (torch.sqrt(squared_norm) + 1e-8)

class GenCapsules(nn.Module):

	def __init__(self, in_caps, num_caps, in_dim, dim_caps, dim_real):

		super(GenCapsules, self).__init__()
		self.dim_real = dim_real
		self.W1 = nn.Parameter(torch.randn(1, 6*6*32, num_caps, in_dim, dim_caps)*(3/(in_dim + dim_caps + 6*6*32))**0.5)
		self.W0 = nn.Parameter(torch.randn(1, num_caps, 1, dim_caps, dim_real)*(3/(dim_caps + num_caps + dim_real))**0.5)



		self.dconv1 = nn.ConvTranspose2d(256, 1, 9, 1, 0)
		self.dconv0 = nn.ConvTranspose2d(256, 256, 10, 2, 0)
		torch.nn.init.xavier_normal_(self.dconv1.weight)
		torch.nn.init.xavier_normal_(self.dconv0.weight)
		self.conv1 = nn.Conv2d(256,256, 1)
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()

		self.batchnorm = nn.BatchNorm2d(256)
		self.batchnorm0 = nn.BatchNorm2d(256)

	def forward(self, x, epoch):

		batch_size = x.size()[0]

		# classes capsules
		#b02 = torch.matmul(self.w0b2, x.unsqueeze(-1))
		#x = (self.w0b1 + 1) * x
		# W0 @ real_struc =
		# (1, num_caps, in_caps, dim_caps, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
		# (batch_size, num_caps, in_caps, dim_caps, 1)
		x = x.unsqueeze(1).unsqueeze(-1)
		c_caps = torch.matmul(self.W0, x)

		# (batch_size, num_caps, in_caps, dim_caps)
		c_caps = c_caps.transpose(1,2).transpose(-2,-1)




		# squash
		# (batch_size, 10, 16)
		c_caps = squash(c_caps).transpose(-2,-1)


		#c_ca = ((self.w1b1 + 1) * c_caps).unsqueeze(-1)

		# # weights
		# w = torch.norm(c_caps, dim=-1)
		# w = F.softmax(w, dim=2).unsqueeze(-1)
		# c_caps = c_caps * w


		# primary capsules

		# W1 @ c_caps =
		# (1, in_caps, num_caps, in_dim, dim_caps) @ (batch_size, 1, in_caps, in_dim, 1) =
		# (batch_size, in_caps, num_caps, in_dim, 1)
		p_caps = torch.matmul(self.W1, c_caps)
		c_caps = c_caps.squeeze(1).squeeze(-1)

		# (batch_size, in_caps, num_caps, in_dim)
		p_caps = p_caps.squeeze(-1)

		#w = torch.norm(p_caps, dim=-1)
		#m = torch.max(w, dim=-1, keepdim=True)[0]
		#mask = w.eq(m).unsqueeze(-1).float()
		#p_caps = p_caps * mask


		# sum projected vectors
		p_caps = p_caps.sum(dim=2)

		# squash
		p_caps = squash(p_caps)


		# reshape capsules for convolutional operations
		# (batch_size, in_caps, in_dim) -> (batch_size, 32, 6, 6, 8) -> (batch_size, 256, 6, 6)
		out = p_caps.view(p_caps.size(0), 32, 6, 6, 8)
		out = out.view(p_caps.size(0), 256, 6, 6)


		# add one layer
		#out = self.conv1(out)
		#out = self.batchnorm0(out)

		# apply deconvs
		out = self.dconv0(out)
		out = self.batchnorm(out)
		out = self.relu(out)

		out = self.dconv1(out)
		out = self.tanh(out)

		return out, p_caps, c_caps
