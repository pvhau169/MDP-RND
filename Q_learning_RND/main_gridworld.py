import numpy as np
import gym
from QRND import QRND
import matplotlib.pyplot as plt
import torch
import gym_gridworld


# file_path = 'plan1.txt'
env = gym.make('gridworld-v0')

gamma = 0.95
timer = 20
scale_intrinsic = 6

alg = QRND(env, gamma, timer, 10000, scale_intrinsic)
# temp = torch.Tensor(env.get_start_state()).unsqueeze(0)
num_epochs = 15000
for i in range(num_epochs):
    log = alg.runEpoch()
    print("epoch ", i, " get return ", np.round(log.get_current('real_return')))
    
PATH = 'model_weights/model_grid_world_1.pth'
torch.save(alg.model.state_dict(), PATH)

Y = np.asarray(log.get_log('real_return'))

log_PATH = 'log/real_return_grid_world_1.txt'
np.savetxt(log_PATH, Y)
# Y2 = smooth(Y)
# x = np.linspace(0, len(Y), len(Y))
# fig1 = plt.figure()
# ax1 = plt.axes()
# ax1.plot(x, Y, Y2)

Y = np.asarray(log.get_log('combined_return'))

log_PATH = 'log/combined_return_grid_world_1.txt'
np.savetxt(log_PATH, Y)
# Y = np.asarray(log.get_log('combined_return'))
# Y2 = smooth(Y)
# x = np.linspace(0, len(Y), len(Y))
# fig2 = plt.figure()
# ax2 = plt.axes()
# ax2.plot(x, Y, Y2)

# obs = env.reset()
# for t in range(1000):
#     env.render()
#     x = torch.Tensor(obs).unsqueeze(0)
#     Q = alg.model(x)
#     action = Q.argmax().detach().item()
#     new_obs, reward, done, info = env.step(action)
#     obs = new_obs
#     if done:
#         break
