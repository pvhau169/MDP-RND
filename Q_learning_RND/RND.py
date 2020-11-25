import torch
import torch.nn

class NeuralNet(torch.nn.Module):
	def __init__(self, in_dim, out_dim, n_hid):
		super(NeuralNet, self).__init__()

		self.in_dim = in_dim
		self.out_dim = out_dim
		self.n_hid= n_hid

		self.layer1 = torch.nn.Linear(in_dim, n_hid, 'linear')
		self.layer2 = torch.nn.Linear(n_hid, n_hid, 'linear')
		self.layer3 = torch.nn.Linear(n_hid, out_dim, 'linear')

		self.softmax = torch.nn.Softmax(dim = 1)


	def forward(self, x):
		x = torch.nn.functional.relu(self.layer1(x))
		x = torch.nn.functional.relu(self.layer2(x))
		y = self.layer3(x)
		# y = self.softmax(y)

		return y

class RandomNetwork:
	def __init__(self, in_dim, out_dim, n_hid):

		self.target = NeuralNet(in_dim, out_dim, n_hid)
		self.model = NeuralNet(in_dim, out_dim, n_hid)

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0001)


	def getReward(self, x):
		y_true = self.target(x).detach()
		y_pred = self.model(x)
		reward = torch.pow(y_pred - y_true, 2).sum()
		return reward
    
	def update(self, reward_i):
		reward_i.sum().backward()
		self.optimizer.step()