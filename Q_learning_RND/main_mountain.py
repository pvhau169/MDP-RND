import numpy as np
import gym
from QRND import QRND
import matplotlib.pyplot as plt
import torch


env = gym.make('MountainCar-v0')

gamma = 0.95
timer = 200
scale_intrinsic = 5
alg = QRND(env, gamma, timer, 10000, scale_intrinsic)


num_epochs = 300
for i in range(num_epochs):
    log = alg.runEpoch()
    print("epoch ", i, " get return ", np.round(log.get_current('real_return')))
    
PATH = 'model_weights/model_mountain_extrinsic_only.pth'
torch.save(alg.model.state_dict(), PATH)

Y = np.asarray(log.get_log('real_return'))

log_PATH = 'log/real_return_mountain_extrinsic_only.txt'
np.savetxt(log_PATH, Y)

Y = np.asarray(log.get_log('combined_return'))

log_PATH = 'log/combined_return_mountain_extrinsic_only.txt'
np.savetxt(log_PATH, Y)

observations = env.reset()
for t in range(1000):
    env.render()
    x = torch.Tensor(observations).unsqueeze(0)
    Q = alg.model(x)
    action = Q.argmax().detach().item()
    new_observations, reward, done, info = env.step(action)
    observations = new_observations
    if done:
        break

