#!/usr/bin/env python3

import gym
import numpy as np

env = gym.make('Taxi-v2')

Q = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.618

for episode in range(1, 1001):
    done = False
    G, reward = 0, 0
    state = env.reset()
    while not done:
        act = np.argmax(Q[state])
        state_, reward, done, info = env.step(act)
        Q[state, act] += alpha * (reward + np.max(Q[state_]) - Q[state, act])
        G += reward
        state = state_
