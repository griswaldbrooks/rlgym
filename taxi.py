#!/usr/bin/env python3

import gym
import numpy as np

env = gym.make('Taxi-v2')

Q = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.618

try:
    for episode in range(1, 1001):
        done = False
        G, reward = 0, 0
        s = env.reset()
        while not done:
            act = np.argmax(Q[s])
            s_, reward, done, info = env.step(act)
            Q[s, act] += alpha * (reward + np.max(Q[s_]) - Q[s, act])
            G += reward
            s = s_
            env.render()
except Exception as e:
    print('oops')

done = False
state = env.reset()
while not done:
    act = np.argmax(Q[state])
    state, r, done, i = env.step(act)
    env.render()
