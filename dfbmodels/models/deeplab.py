import torch

import numpy as np

import torchvision

from torchvision.models.resnet import BasicBlock, Bottleneck

import segmentation_models_pytorch as smp

from dfbmodels.equilibrium import EquilibriumModel

from dfbmodels.state import StateExpDecay#StateConvExpDecay

class DeepLabV3Plus(smp.DeepLabV3, EquilibriumModel):

	def __init__(self, in_filters=3, fb_filters=2, class_filters=256, registers=0, classes=8, alpha=0.2, tau=1e0, shape=(512,512), block=BasicBlock, layers=[2,2,2,2]):
		super(DeepLabV3Plus, self,).__init__(in_channels=in_filters, decoder_channels=class_filters+fb_filters, classes=classes)

		self.in_filters=in_filters
		self.fb_filters=fb_filters
		self.class_filters=class_filters

		self.frontend=torch.nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)
		self.maxpool=self.encoder.maxpool
		
		del self.encoder, self.decoder, self.segmentation_head
		
		#create resnet18 as encoder
		self.expansion=block.expansion
		self.encoder=resnet.ResNetEncoder(in_channels=64+fb_filters, **{"out_channels": (3, 64+fb_filters, 64+fb_filters, 128, 256, 512),"block": block,"layers": layers,})
		encoder_channels=[3,64,64*self.expansion,128*self.expansion,256*self.expansion,512*self.expansion]
		self.decoder=smp.decoders.deeplabv3.decoder.DeepLabV3PlusDecoder(encoder_channels=encoder_channels,out_channels=fb_filters+class_filters,atrous_rates=(12, 24, 36),encoder_depth=5,output_stride=16,aspp_separable=True,aspp_dropout=0.5)
		self.segmentation_head=smp.base.SegmentationHead(in_channels=class_filters,out_channels=classes,activation=None,kernel_size=1,upsampling=4,)
		
		smp.base.initialization.initialize_head(self.segmentation_head)
		
		self.shape=(shape[0]//4,shape[1]//4)

		self.h_class=StateExpDecay(filters=class_filters+fb_filters+registers, shape=self.shape, tau=1e0)
		self.noise=None
		
		self.T=torch.nn.Parameter(torch.zeros(()), requires_grad=False)

		self.loss_fn=torch.nn.functional.cross_entropy
		
		self.ground_truth=None

		self.compute_softmax=True
		
		self.compute_error=True
		if(self.compute_error):
			self.error_head=StateExpDecay(class_filters+fb_filters+registers,shape=self.shape,tau=tau)
			#self.error_head=StateConvExpDecay(class_filters+fb_filters+registers,3,stride=1,tau=tau)

		self.layernorm_error=torch.nn.GroupNorm(num_groups=256+fb_filters, num_channels=256+fb_filters)#torch.nn.LayerNorm((2,2),)#torch.nn.Identity()#
		self.layernorm_input=torch.nn.GroupNorm(num_groups=fb_filters, num_channels=fb_filters)#torch.nn.LayerNorm((self.shape[0]*2, self.shape[1]*2),)

		torch.nn.init.constant_(self.layernorm_error.weight, 1e0)
		torch.nn.init.constant_(self.layernorm_input.weight, 1e0)
		torch.nn.init.constant_(self.layernorm_error.bias, 0)
		torch.nn.init.constant_(self.layernorm_input.bias, 0)
		
		self.layernorm_error.weight.requires_grad=False
		self.layernorm_error.bias.requires_grad=False
		self.layernorm_input.weight.requires_grad=False
		self.layernorm_input.bias.requires_grad=False

		self.verbosity=False

		self.alpha=torch.nn.Parameter(torch.tensor(alpha), requires_grad=False)
		self.one=torch.nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)
		self.dt=torch.nn.Parameter(torch.tensor(1./5.), requires_grad=False)
		self.Tf=torch.nn.Parameter(torch.tensor(5.), requires_grad=False)

	def initT(self, T):
		torch.nn.init.constant_(self.dt, 1./(T))
		torch.nn.init.constant_(self.Tf, (T))

	@property
	def states(self,):
		return {"h": self.h_class.state}

	def prediction(self,):
		return self.segmentation_head(self.h_class.state[:,:self.class_filters])

	def step(self, x):
		
		t=self.T

		ht=self.h_class.state[:,self.class_filters:self.class_filters+self.fb_filters]
		ht=torch.nn.functional.interpolate(ht, size=(self.shape[0]*2,self.shape[1]*2), mode="bilinear")
		ht=self.layernorm_input(ht)
		if(self.compute_softmax):
			ht=torch.nn.functional.softmax(ht, dim=1)
		xtilde=torch.concat((x, ht), dim=1)

		#function
		xtilde=self.maxpool(xtilde)
		features = self.encoder(xtilde)
		y=self.decoder([None, None, *features])

		#nosie
		torch.nn.init.zeros_(self.noise,)
		#torch.nn.init.normal_(self.noise, mean=0., std=1e-6)

		y=self.layernorm_error(y)+self.noise
		if(self.compute_error): y=self.error_head(y, T=self.T)#*(self.Tf*self.dt-t*self.dt)**(self.alpha-1)/torch.exp(torch.special.gammaln(self.alpha))

		self.h_class.state=self.h_class.state+y

		torch.nn.init.constant_(self.T, self.T+1)
			
	def forward(self, x, T=0, atol=1e-2):
		"""Sequentially pass `x` trough model`s encoder, decoder and heads"""

		x=self.frontend(x)#B \times C \times 16 \times 16
		
		torch.nn.init.zeros_(self.T)

		#initialize states
		self.init_states()
		self.h_class.state=torch.zeros((x.shape[0], *self.h_class.state.shape[1:]), dtype=x.dtype, device=x.device)
		self.noise=torch.zeros((x.shape[0], *self.h_class.state.shape[1:]), dtype=x.dtype, device=x.device)
		torch.nn.init.normal_(self.h_class.state, mean=0, std=1e-3)
		
		if(T==0): return self.prediction()
		
		self.initT(T)
		
		self.to_equilibrium(x, None, None, T=T-1, atol=atol)#we de not detach x so gradients run to the first layer

		self.step(x)#when T=1, it runs this step only
		
		return self.prediction()