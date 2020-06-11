#!/usr/bin/env python

"""
Builds a convolutional neural network on the fashion mnist data set.

Designed to show wandb integration with pytorch.
"""

import wandb
import spinup
import gym
import pybullet_envs


hyperparameter_defaults = dict(
    coeff = 0.5,
    lam1 = 0.93, 
    lam2 = 0.97,
    env_name = 'AntBulletEnv-v0'
    )


if __name__ == '__main__':
   wandb.init(config=hyperparameter_defaults, project='generalized-critic', entity='syrma', monitor_gym=True)
   config = dict(wandb.config)

   # Make 'env_fn' from 'env_name'
   if 'env_name' in config:
      env_name = config['env_name']
      config['env_fn'] = lambda : gym.make(env_name)
      del config['env_name']

   # Run thunk
   spinup.ppo2_tf1(**config)
