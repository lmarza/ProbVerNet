import torch

# Torch class for the scalability test
class ScalabilityNet(torch.nn.Module):
	def __init__(self, input_size, output_size):
		super(ScalabilityNet, self).__init__()
		self.fc1 = torch.nn.Linear(input_size, 32)
		self.fc2 = torch.nn.Linear(32, 32)
		self.fc3 = torch.nn.Linear(32, output_size)

	def forward(self, x):
		x = torch.torch.nn.functional.relu(self.fc1(x))
		x = torch.torch.nn.functional.relu(self.fc2(x))
		x = self.fc3(x)
		return x


# Torch class for the ACAS Xu test (Property 2)
class AcasNetP2(torch.nn.Module):
	def __init__(self, input_size, output_size):
		super(AcasNetP2, self).__init__()
		self.fc1 = torch.nn.Linear(input_size, 50)
		self.fc2 = torch.nn.Linear(50, 50)
		self.fc3 = torch.nn.Linear(50, 50)
		self.fc4 = torch.nn.Linear(50, 50)
		self.fc5 = torch.nn.Linear(50, 50)
		self.fc6 = torch.nn.Linear(50, 50)
		self.fc7 = torch.nn.Linear(50, output_size)

	def forward(self, x, real_output=False):
		x = torch.torch.nn.functional.relu(self.fc1(x))
		x = torch.torch.nn.functional.relu(self.fc2(x))
		x = torch.torch.nn.functional.relu(self.fc3(x))
		x = torch.torch.nn.functional.relu(self.fc4(x))
		x = torch.torch.nn.functional.relu(self.fc5(x))
		x = torch.torch.nn.functional.relu(self.fc6(x))
		x = self.fc7(x)

		if real_output: return x
		
		res = [1 if el else -1 for el in torch.argmax(x, dim=1).numpy() != 0]
		return torch.tensor( res )  
		

# Torch class for the ACAS Xu test (Property 3)
class AcasNetP3(torch.nn.Module):
	def __init__(self, input_size, output_size):
		super(AcasNetP3, self).__init__()
		self.fc1 = torch.nn.Linear(input_size, 50)
		self.fc2 = torch.nn.Linear(50, 50)
		self.fc3 = torch.nn.Linear(50, 50)
		self.fc4 = torch.nn.Linear(50, 50)
		self.fc5 = torch.nn.Linear(50, 50)
		self.fc6 = torch.nn.Linear(50, 50)
		self.fc7 = torch.nn.Linear(50, output_size)

	def forward(self, x, real_output=False):
		x = torch.torch.nn.functional.relu(self.fc1(x))
		x = torch.torch.nn.functional.relu(self.fc2(x))
		x = torch.torch.nn.functional.relu(self.fc3(x))
		x = torch.torch.nn.functional.relu(self.fc4(x))
		x = torch.torch.nn.functional.relu(self.fc5(x))
		x = torch.torch.nn.functional.relu(self.fc6(x))
		x = self.fc7(x)

		if real_output: return x

		res = [1 if el else -1 for el in torch.argmin(x, dim=1).numpy() != 0]
		return torch.tensor( res )  
	

# Torch class for the TurtleBot test (Property 'front')
class TurtleNetFront(torch.nn.Module):
	def __init__(self, input_size, output_size):
		super(TurtleNetFront, self).__init__()
		self.fc1 = torch.nn.Linear(input_size, 64)
		self.fc2 = torch.nn.Linear(64, 64)
		self.fc3 = torch.nn.Linear(64, output_size)

	def forward(self, x, real_output=False):
		x = torch.torch.nn.functional.relu(self.fc1(x))
		x = torch.torch.nn.functional.relu(self.fc2(x))
		x = torch.torch.nn.functional.softmax(self.fc3(x))

		if real_output: return x
		
		res = [1 if el else -1 for el in torch.argmax(x, dim=1).numpy() != 4]
		return torch.tensor( res )  
		