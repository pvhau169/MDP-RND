import numpy as np
import gym
from QRND import QRND
import matplotlib.pyplot as plt
import torch


env = gym.make('MountainCar-v0')
PATH = 'model_weights/model.pth'

gamma = 0.95
timer = 200
scale_intrinsic = 5
alg = QRND(env, gamma, timer, 10000, scale_intrinsic)
alg.model.load_state_dict(torch.load(PATH))

obs = env.reset()

for t in range(1000):
    env.render()
    x = torch.Tensor(obs).unsqueeze(0)
    Q = alg.model(x)
    action = Q.argmax().detach().item()
    new_obs, reward, done, info = env.step(action)
    obs = new_obs
    if done:
        break