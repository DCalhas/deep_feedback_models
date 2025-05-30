import torch

import numpy as np

class EquilibriumModel(torch.nn.Module):
	"""
	This class implements fixed point iteration
	"""
	def init_states(self,):
		#initialize beliefs
		for key in self.states.keys():
			torch.nn.init.zeros_(self.states[key])
	
	@property
	def states(self,):
		"""
		This method should be implemneted by the subclass
		"""
		raise NotImplementedError

	def step(self, x):
		"""
		This method should be implemented by the subclass

		A step in differs from system to system
		"""
		raise NotImplementedError

	def prediction(self,):
		"""
		This method should be implemented by the subclass

		This returns the prediction given current state of the model
		"""
		raise NotImplementedError

	def to_equilibrium(self, ims, masks, loss_fn, T=None, atol=1e-2):
		"""
		Fixed point iteration to equilibrium state
		"""

		if(T is not None and T <= 0):
			return 
		
		self.init_states()
	
		if T is None:
			T = 2**32 - 1
		prev = float('inf')
	
		for _ in range(T):
			out = self.step(ims)
			eq = sum(state.abs().sum() for state in self.states.values())
			eq_val = eq.item()
			if abs(eq_val - prev) <= atol:
				break
			prev = eq_val