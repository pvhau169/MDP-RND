import torch 
import numpy as np
from RND import RandomNetwork
import random
import copy
# import torch.nn.functional as F
from collections import deque
from log_utils import logger, mean_val


class NeuralNet(torch.nn.Module):
	def __init__(self, in_dim, out_dim, n_hid):
		super(NeuralNet, self).__init__()

		self.in_dim = in_dim
		self.out_dim = out_dim
		self.n_hid= n_hid

		self.layer1 = torch.nn.Linear(in_dim, n_hid, 'relu')
		# self.layer2 = torch.nn.Linear(n_hid, n_hid, 'linear')
		self.layer3 = torch.nn.Linear(n_hid, out_dim, 'linear')

		# self.softmax = torch.nn.softmax(dim = 1)


	def forward(self, x):
		x = torch.nn.functional.relu(self.layer1(x))
		# x = torch.nn.functional.relu(self.layer2(x))
		y = self.layer3(x)
		# y = self.softmax(y)

		return y

class QRND:
	def __init__(self,  env,  gamma,  timer,  buffer_size, scale_intrinsic):
		self.env = env
		actions = env.action_space
		observations = env.observation_space
		# observations = np.ones((2,1))
		# observations = np.ones((2, 1))
		self.timer = timer
		self.gamma = gamma
		self.buffer_size = buffer_size
		self.scale_intrinsic = scale_intrinsic

		self.model = NeuralNet(observations.shape[0], actions.n, 64)
		self.target_model = copy.deepcopy(self.model)
		self.rnd = RandomNetwork(observations.shape[0], 64, 124)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)


		self.batch_size = 64
		
		self.epsi_high = 0.9
		# self.epsilon = self.epsi_high
		self.epsi_low = 0.05

		
		self.step_counter = 0
		# self.epsi_high = 0.9
		# self.epsi_low = 0.05
		self.steps = 0
		self.count = 0
		self.decay = 200
		self.eps = self.epsi_high
		# self.eps = 0.3

		self.update_target_step = 300
		self.log = logger()
		self.log.add_log('real_return')
		self.log.add_log('combined_return')
		self.log.add_log('avg_loss')
		
		self.replay_buffer = deque(maxlen=buffer_size)
		

	def runEps(self):
		obs = self.env.reset()

		sum_r = 0
		sum_tot_r = 0
		mean_loss = mean_val()
		
		for t in range(self.timer):
			self.steps += 1
			#decay epsilon
			self.eps = self.epsi_low + (self.epsi_high-self.epsi_low) * (np.exp(-1.0 * self.steps/self.decay))
			state = torch.Tensor(obs).unsqueeze(0)
			Q = self.model(state)
			num = np.random.rand()


			if (num < self.eps):
				action = torch.randint(0, Q.shape[1], (1, )).type(torch.LongTensor)
			else:
				action = torch.argmax(Q, dim=1)


			new_state,  reward,  done,  info = self.env.step(action.item())
			sum_r = sum_r + reward
			reward_i = self.rnd.getReward(state).detach().clamp(-1.0, 1.0).item()
			combined_reward = reward + self.scale_intrinsic * reward_i
			sum_tot_r += combined_reward
			
			self.replay_buffer.append([obs, action, combined_reward, new_state, done])

			loss = self.update()
			mean_loss.append(loss)
			obs = new_state
            
			self.step_counter += 1
			if (self.step_counter > self.update_target_step):
				self.target_model.load_state_dict(self.model.state_dict())
				self.step_counter = 0

			if done:
				break
		self.log.add_item('real_return', sum_r)
		self.log.add_item('combined_return', sum_tot_r)
		self.log.add_item('avg_loss', mean_loss.get())
        
	def update(self):
		self.optimizer.zero_grad()
		num = len(self.replay_buffer)
		K = np.min([num, self.batch_size])

		samples = random.sample(self.replay_buffer,  K)
        
		state,  action,  reward,  next_state,  done = zip(*samples)
		state = torch.tensor(state,  dtype=torch.float)
		action = torch.tensor(action,  dtype=torch.long).view(K,  -1)
		reward = torch.tensor(reward,  dtype=torch.float).view(K,  -1)
		next_state = torch.tensor(next_state,  dtype=torch.float)
		done = torch.tensor(done,  dtype=torch.float)
       
		Ri = self.rnd.getReward(state)
		self.rnd.update(Ri)

		target_q = reward.squeeze() + self.gamma*self.target_model(next_state).max(dim=1)[0].detach()*(1 - done)
		policy_q = self.model(state).gather(1, action)

		L = torch.nn.functional.smooth_l1_loss(policy_q.squeeze(), target_q.squeeze())
		L.backward()
		self.optimizer.step()
		return L.detach().item()
    
	def runEpoch(self):
		self.runEps()
		return self.log
