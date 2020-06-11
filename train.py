#!/usr/bin/env python

"""
Builds a convolutional neural network on the fashion mnist data set.

Designed to show wandb integration with pytorch.
"""

import wandb
import os 
hyperparameter_defaults = dict(
    coeff = 0.5,
    lam1 = 0.93, 
    lam2 = 0.97,
    env = 'AntBulletEnv-v0'
    )

wandb.init(config=hyperparameter_defaults, project='generalized-critic', entity='syrma', monitor_gym=True)
config = wandb.config

def main():
   cmd = 'python -m spinup.run ppo2 --env_name ' + config.env + ' --coeff ' + str(config.coeff)
   os.system(cmd)


if __name__ == '__main__':
   main()
