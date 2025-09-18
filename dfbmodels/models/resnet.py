import torch

from dfbmodels.state import StateExpDecay

from dfbmodels.equilibrium import EquilibriumModel

from torchvision.models.resnet import BasicBlock, Bottleneck

from torchvision.models import ResNet as _ResNet

class ResNet(_ResNet, EquilibriumModel):
	"""
	This model inherits mainly from ResNet and has functionalities taken from EquilibriumModel.
	It is a resnet that performs fixed point iteration of a state until equilibrium. This is done using
	the StateExpDecay class that guarantees the stability of this system.
	Example usage:
	>>> from dfbmodels.layers import ResNet
	>>> model=ResNet(fb_filters=200, n_classes=200, shape=(64,64))
	>>> x=torch.ones((1,3,64,64), dtype=torch.float32)
	>>> model(x).sum().backward()
	"""

	def __init__(self, frontend='resnet', fb_filters=0, registers=0, n_classes=200, shape=(64,64), tau=1e0, block=BasicBlock, layers=[3,4,6,3]):
		"""
		This is the init function of the NeumannResnet class, the arguments of the call are:
		* frontend - str - specifies the frontend defined in this model, for now we are just implementing the standard of the resnet
		* fb_filters - int - specifies the number of feedback neurons that are taken from higher level representation and concatenated with the representation that comes out of the frontend
		* registers - int - specifies the number of neurons that register information, this is always set to zero in our experiments
		* n_classes - int - specifies the number of classes the final layer has to map to
		* shape - tuple(int) - specifies the shape of the input, so we can set the StateExpDecay function
		* tau - float - specifies the tempo that is used to solve the differential equation
		* block - torchvision.models.resnet.Block - the block can be either BasicBlock or Bottleneck, this parameter is given to the Resnet constructor
		* layers - list(int) - specifies the number of blocks per layer, this parameter is given to the Resnet constructor
		
		"""
		super(ResNet, self).__init__(block, layers, num_classes=n_classes)

		nodes=4
		self.shape=shape
		self.nodes=nodes
		self.block_expansion=block.expansion

		del self.layer1, self.layer2, self.layer3, self.layer4

		self.fc=torch.nn.Linear(512 * self.block_expansion, n_classes)

		self.inplanes=64+(fb_filters)*self.block_expansion
		self.layer1=self._make_layer(block,64, layers[0])
		self.layer2=self._make_layer(block,128, layers[1], stride=2)
		self.layer3=self._make_layer(block,256, layers[2], stride=2)
		self.layer4=self._make_layer(block,512+fb_filters, layers[3], stride=2)

		if(frontend=='resnet'):
			in_channels=self.layer1[0].conv1.out_channels
			self.frontend=torch.nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)
		elif(frontend=='evnet'):
			raise NotImplementedError
			self.frontend=torch.nn.Sequential(vone, bottleneck)
		else:
			raise NotImplementedError

		self.initialize()

		self.in_channels=in_channels
		self.fb_filters=fb_filters
		self.registers=registers
		self.classes=n_classes

		#now we can start to implement the feedback rationale
		self.classification_head=torch.nn.Sequential(*[self.avgpool, torch.nn.Flatten(start_dim=1), self.fc])

		self.state=torch.zeros(())

		self.error_head=self.create_error_head(512*self.block_expansion, fb_filters*self.block_expansion, self.shape, tau=tau)
		
		self.layernorm_error=torch.nn.GroupNorm(num_groups=8, num_channels=(512+fb_filters)*self.block_expansion)
		self.layernorm_input=torch.nn.GroupNorm(num_groups=8, num_channels=fb_filters*self.block_expansion)
		
		torch.nn.init.constant_(self.layernorm_error.weight, 1)
		torch.nn.init.constant_(self.layernorm_error.bias, 0)
		torch.nn.init.constant_(self.layernorm_input.weight, 1)
		torch.nn.init.constant_(self.layernorm_input.bias, 0)

		self.layernorm_weight=torch.nn.Parameter(torch.zeros_like(self.layernorm_error.weight).unsqueeze(0).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
		torch.nn.init.zeros_(self.layernorm_weight)
		self.layernorm_error.weight.requires_grad=False
		self.layernorm_error.bias.requires_grad=False
		self.layernorm_input.weight.requires_grad=False
		self.layernorm_input.bias.requires_grad=False

		self.T=torch.nn.Parameter(torch.zeros(()), requires_grad=False)

		self.layer1=torch.compile(self.layer1)
		self.layer2=torch.compile(self.layer2)
		self.layer3=torch.compile(self.layer3)
		self.layer4=torch.compile(self.layer4)
		self.frontend=torch.compile(self.frontend)

	def create_error_head(self, class_filters, fb_filters, shape, tau=1e0):
		"""
		This function creates the error head, which for this architecture is always the StateExpDecay class.
		It returns an error to correct a state variable. The error decays as $t \to \infty$
		"""
		return StateExpDecay(class_filters+fb_filters, (shape[0]//4,shape[1]//4,), tau=tau)

	def initialize(self,):
		"""
		This is the intialization of the modules of this neural network. This is sourced from https://github.com/qubvel-org/segmentation_models.pytorch/
		"""
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d):
				torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
			elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
				torch.nn.init.constant_(m.weight, 1)
				torch.nn.init.constant_(m.bias, 0)
		torch.nn.init.kaiming_normal_(self.fc.weight)
		torch.nn.init.constant_(self.fc.bias, 0)

	def to(self, device="cuda:0", **kwargs):
		"""
		Maps the tensors to the cuda, whenever the module.to method is called.
		"""
		self.state=self.state.to(device=device)
		
		super(ResNet,self).to(device=device, **kwargs)

	@property
	def states(self):
		"""
		The goal in the beginning was to have state variables after each layer, however we decided to alter only the last one.
		"""
		return {"layer4": self.state,  }

	
	@property
	def inlayer1(self,):
		"""
		Returns a representation that should be concatenated with the output of the frontend of the resnet.
		This is the softmax of the upsampled representation that emulates the output of layer4. Note that this representation state["layer4"]
		is the one that is used to make the prediction as you will see in the prediction method of this class.
		We apply layer norm and softmax in order to make the system stable.
		"""
		states=self.states

		return torch.softmax(self.layernorm_input(torch.nn.functional.interpolate(states["layer4"][:, 512*self.block_expansion:(512+self.fb_filters)*self.block_expansion], size=(self.shape[0]//4, self.shape[1]//4), mode='bilinear')), dim=1)
	
	def _init_states(self, x):
		"""
		This function initializes the hidden states of this recurrent neural network that uses the Resnet as a cell
		The initialization is performed with a normal distribution $\mathcal{N}(0, 0.001)$
		"""
		self.state=torch.ones((x.shape[0], (512+self.fb_filters)*self.block_expansion, x.shape[2]//8, x.shape[2]//8), device=x.device)
		torch.nn.init.normal_(self.state, mean=0., std=1e-3)
	
	def step(self, x):
		"""
		This is the step function, that solves the differential equation $\frac{\mbox{d} \mathbf{h}}{\mbox{d}t} = F \left(\left[ \mathbf{x}, \frac{\mbox{exp}\left(\hinput(t)\right)}{\sum_i \mbox{exp}\left(\hinput_i(t)\right)} \right]; \theta \right)$
		"""

		t=self.T
		
		x=self.layer1(torch.concat((x, self.inlayer1), dim=1))
		x=self.layer2(x)
		x=self.layer3(x)
		x=self.layer4(x)
		
		#maybe remove the fractional component
		self.state=self.state+self.error_head(self.layernorm_error(x), T=self.T)

		torch.nn.init.constant_(self.T, self.T+1)
	
	@torch.compile	
	def prediction(self,):
		"""
		This function filters the hidden state right part of the vector to feed to the classification head to get a decision.
		The mathematical representation of this is $\hat{\mathbf{y}}(\mathbf{h}, t) = G\left( \houtput(t) \right)$.
		"""
		return self.classification_head(self.state[:,:512*self.block_expansion])#return for prediction

	def forward(self, x, T=0, atol=1e-2):
		"""
		This is the forward pass, which starts by feeding x to the frontend that is defined,
		then we initialize the variables needed to solve the differential equation.
		If T is greater than zero than we call to_equilibrium to perform fixed point iteration and return the prediction
		"""

		x=self.frontend(x)#B \times C \times 16 \times 16
		
		torch.nn.init.zeros_(self.T)

		#initialize states
		self._init_states(x)
		
		if(T==0): return self.prediction()
		
		self.to_equilibrium(x, None, None, T=T-1, atol=atol)#we de not detach x so gradients run to the first layer

		self.step(x)#when T=1, it runs this step only
		
		return self.prediction()

class ResNet18(ResNet):

	def __init__(self, frontend='resnet', fb_filters=0, registers=0, n_classes=200, shape=(64,64), tau=1e0):
		super(ResNet18, self).__init__(frontend=frontend, fb_filters=fb_filters, registers=registers, n_classes=n_classes, shape=shape, tau=tau, block=BasicBlock, layers=[2,2,2,2])

class ResNet34(ResNet):

	def __init__(self, frontend='resnet', fb_filters=0, registers=0, n_classes=200, shape=(64,64), tau=1e0):
		super(ResNet34, self).__init__(frontend=frontend, fb_filters=fb_filters, registers=registers, n_classes=n_classes, shape=shape, tau=tau, block=BasicBlock, layers=[3,4,6,3])

class ResNet50(ResNet):

	def __init__(self, frontend='resnet', fb_filters=0, registers=0, n_classes=200, shape=(224,224), tau=1e0):
		super(ResNet50, self).__init__(frontend=frontend, fb_filters=fb_filters, registers=registers, n_classes=n_classes, shape=shape, tau=tau, block=Bottleneck, layers=[3,4,6,3])