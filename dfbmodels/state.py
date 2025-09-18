import torch

import numpy as np

class ExpDecay(torch.autograd.Function):
	"""
	This class implements exponential decay given a time variable
	$\frac{\mbox{d} \mathbf{h}}{\mbox{d}t} = \mathbf{h}(t) \cdot \Sigma$
	This makes the forward pass be defined as 
	$\mathbf{h}(t)=\mathbf{h}(0) \cdot e^{-\tau^{-1}t\Sigma}$
	However the input of this layer, is represented above as $\mathbf{h}(0)$, so it can be any variable.
	"""
	@staticmethod
	def step(x, T, Delta):
		"""
		This function represents a forward step of size $T$ and multiplies the input with the exponentiated kernel.
		"""
		exp_term = torch.exp(T * Delta).unsqueeze(0).unsqueeze(2).unsqueeze(3)  # Shape: (1, Delta.size, 1, 1)
		return x * exp_term  # Broadcasting over x dimensions

	@staticmethod
	def backwardH(grad, Delta, T):
		"""
		This is the backward pass, the derivative w.r.t. x, which is just the kernel.
		"""
		exp_term = torch.exp(Delta * T).unsqueeze(0).unsqueeze(2).unsqueeze(3)  # times T is needed to avoid gradient explosion
		return grad * exp_term

	@staticmethod
	def backwardDelta(grad, x, Delta, T):
		"""
		This is the backward pass, the derivative w.r.t. Delta, which is the exponential times the input x. T is omitted.
		"""
		exp_term = torch.exp(Delta * T).unsqueeze(0).unsqueeze(2).unsqueeze(3)  # times T is needed to avoid gradient explosion
		result = (grad * x * exp_term).mean(dim=(0, 1, 2))  # Reduce across batch and spatial dimensions
		return result

	@staticmethod
	def forward(ctx, x, T, Delta):
		"""
		We perform a step in the forward pass and save the needed tensors for backward
		"""
		z = ExpDecay.step(x, T, Delta)
		ctx.save_for_backward(x, T, Delta, z)
		return z

	@staticmethod
	def backward(ctx, grad_output):
		"""
		The raw implementation of the backward pass. Since we do not train neither the Delta nor the T we return None gradients for these tensors
		We do return the gradient for X.
		"""
		x, T, Delta, z = ctx.saved_tensors
		
		# Compute gradients
		gradX = ExpDecay.backwardH(grad_output, Delta, T)  # Gradient w.r.t. x

		return gradX, None, None


class StateExpDecay(torch.nn.Module):
	"""
	The StateExpDecay is a torch module that performs exponential decay on a state.
	It is up to the user to update the state attribute outside this class.
	Example usage:
		>>> layer=StateExpDecay(64, (64, 512, 512,), tau=0.5)
		>>> x=x.to(device="cuda:0")
		>>> layer.to(device="cuda:0")
		>>> z=layer(x)
		>>> z.sum().backward()
		>>> 
		>>> print(z.shape)
	"""
	def reorthogonalize(p):
		"""
		This is the QR decomposition correction made to a parameter p
		Consider $p=Q \cdot R@$, where $Q$ is orthogonal and $R$ is an upper triangular matrix.
		A call to this function sets Q to the parameter given.
		"""
		with torch.no_grad(): 
			p.copy_(torch.linalg.qr(p, mode="complete")[0])

	def __init__(self, filters, shape, tau=0.1, alpha=1.):
		"""
		This function sets the parameters of the StateExpDecay, these are:
		* filters - int - channels dimension
		* tau - torch.Parameter - tempo
		* T - torch.Parameter - time variable
		* alpha - torch.Parameter - exponentiates the eigenvalues
		* state - torch.tensor - saves the state that is up for the user to update outside of this class
		* S - torch.Parameter - represents the eigenvalues of this transformation
		* Q - torch.Parameter - represents the eigenvectors of this transformation
		* Qinv - torch.Parameter - represents the inverse of the eigenvectors, this is redundant if one uses QR correction
		* autograd_fn - torch.autograd.Function
		"""
		super(StateExpDecay, self).__init__()

		self.filters=filters

		self.tau=torch.nn.Parameter(tau*torch.ones((),), requires_grad=False)
		self.T=torch.nn.Parameter(torch.ones((),), requires_grad=False)
		self.alpha=torch.nn.Parameter(alpha*torch.ones((),), requires_grad=False)
		self.state=torch.zeros((1,filters, *shape), requires_grad=False)
		
		self.S=torch.nn.Parameter(torch.ones((filters,), dtype=torch.float32), requires_grad=False)
		self.Q=torch.nn.Parameter(torch.ones((filters,filters), dtype=torch.float32), requires_grad=True)
		self.Qinv=torch.nn.Parameter(torch.ones((filters,filters), dtype=torch.float32), requires_grad=True)

		torch.nn.init.ones_(self.S)
		torch.nn.init.orthogonal_(self.Q)
		torch.nn.init.orthogonal_(self.Qinv)
		
		self.autograd_fn=ExpDecay

	def solve(self, x, S, Q, Qinv, T,):
		"""
		This is a torch function that maps x to a spectral space via $x \cdot Q$, then
		it performs exponential decay with the self.autograd_fn that basically does $\cdot e^{-\tau^{-1} t \Sigma}$
		and finally it maps it back to the original space via $\cdot Q^{-1}$
		"""
		z=torch.nn.functional.linear(x.permute(0,2,3,1), Q).permute(0,3,1,2,)
		
		z2=self.autograd_fn.apply(z, self.T*T, S)

		return torch.nn.functional.linear(z2.permute(0,2,3,1), Qinv).permute(0,3,1,2,)

	def forward(self, x, T=5):
		"""
		The forward pass of this module first verifies if it is training and it is the first timestep, we do reorthogonalization
		After we ensure that $Q$ is orthogonal, then we can solve this differential equation
		"""
		if(self.training and T==0):
			StateExpDecay.reorthogonalize(self.Q)
		
		return self.solve(x, (1/self.tau)*-1.*torch.abs(self.S)**self.alpha, self.Q, self.Q.T, T)

class StateConvExpDecay(torch.nn.Module):
	"""
	The StateConvExpDecay is a torch module that performs exponential decay on a state.
	The difference between this class and StateExpDecay is that here we perform exponential decay locally,
	by applying decay on a kernel that slides through the x representation.
	One of the advantages of this layer is that it does not need the shape of the input.
	Similarly to StateExpDecay, it is up to the user to update the state attribute outside this class.
	Example usage:
		>>> layer=StateConvExpDecay(64, (64, 512, 512,), tau=0.5)
		>>> x=x.to(device="cuda:0")
		>>> layer.to(device="cuda:0")
		>>> z=layer(x)
		>>> z.sum().backward()
		>>> 
		>>> print(z.shape)
	"""
	def reorthogonalize(p):
		"""
		This is the QR decomposition correction made to a parameter p
		Consider $p=Q \cdot R@$, where $Q$ is orthogonal and $R$ is an upper triangular matrix.
		A call to this function sets Q to the parameter given.
		"""
		assert len(p.size())==2

		with torch.no_grad():
			if(p.shape[0]<p.shape[1]):
				p.copy_(torch.linalg.qr(p.T, mode="reduced")[0].T)
			else:
				p.copy_(torch.linalg.qr(p, mode="complete")[0])

	def __init__(self, filters, kernel_size, stride=1, tau=0.1, alpha=1.):
		"""
		This function sets the parameters of the StateConvExpDecay, these are:
		* filters - int - channels dimension
		* tau - torch.Parameter - tempo
		* T - torch.Parameter - time variable
		* alpha - torch.Parameter - exponentiates the eigenvalues
		* state - torch.tensor - saves the state that is up for the user to update outside of this class
		* S - torch.Parameter - represents the eigenvalues of this transformation
		* U - torch.Parameter - represents the left eigenvectors of this transformation
		* V - torch.Parameter - represents the right eigenvectors of this transformation
		* Q - torch.Parameter - represents the eigenvectors of this transformation, it is not used here, it is just a placeholder
		* autograd_fn - torch.autograd.Function
		"""
		super(StateConvExpDecay, self).__init__()
	
		self.filters=filters
		self.kernel_size=kernel_size
		self.stride=stride

		self.tau=torch.nn.Parameter(tau*torch.ones((),), requires_grad=False)
		self.T=torch.nn.Parameter(torch.ones((),), requires_grad=False)
		self.alpha=torch.nn.Parameter(alpha*torch.ones((),), requires_grad=False)
		
		self.S=torch.nn.Parameter(torch.ones((filters,), dtype=torch.float32), requires_grad=False)
		self.U=torch.nn.Parameter(torch.ones((filters,filters), dtype=torch.float32), requires_grad=True)
		self.V=torch.nn.Parameter(torch.ones((filters, filters*kernel_size*kernel_size), dtype=torch.float32), requires_grad=True)

		#placeholder until deprecation
		self.Q=torch.nn.Parameter(torch.ones((filters,filters,kernel_size,kernel_size), dtype=torch.float32), requires_grad=True)

		torch.nn.init.ones_(self.S)
		torch.nn.init.orthogonal_(self.U)
		torch.nn.init.orthogonal_(self.V)

		self.autograd_fn=layers.state.ExpDecay

		#run deterministically
		assert os.environ["CUBLAS_WORKSPACE_CONFIG"]==":4096:8" or os.environ["CUBLAS_WORKSPACE_CONFIG"]==":16:8"
		torch.use_deterministic_algorithms(True, warn_only=True)

	def solve(self, x, S, Q, Qinv, T,):
		"""
		This is very similar to the solve of StateExpDecay, however here we compute the kernel first and then use it
		to dissipate energy in x. Normalization by the size of the kernel is needed in order to avoid gradient explosion
		"""
		z=torch.nn.functional.linear(x.permute(0,2,3,1), Q).permute(0,3,1,2,)

		z2=self.autograd_fn.apply(z, self.T*T, S)
		
		return torch.nn.functional.conv2d(z2, Qinv.reshape(self.filters, self.filters, self.kernel_size, self.kernel_size), stride=self.stride, padding="same")/self.kernel_size
		
	def forward(self, x, T=5):
		"""
		The forward pass first corrects U and V to be orthogonal, if the model is training and it is the first step taken in decay.
		After correction is done, the differential equation is solved.
		"""
		if(self.training and T==0):
			StateConvExpDecay.reorthogonalize(self.U)
			StateConvExpDecay.reorthogonalize(self.V)

		return self.solve(x, -1*(1/self.tau)*self.S**self.alpha, self.U, self.V, T)
