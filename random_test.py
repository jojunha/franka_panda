import numpy as np
import frankaEnv

env = frankaEnv.FrankaEnv()
env._max_episode_steps = 500 # setup max_ steps per episode

def main():

    while True:
        obs = env.reset()
        episode_reward = 0
        
        for _ in range(10000):
            action = env.action_space.sample()  
            obs, rew, dones, info = env.step(action, render = True)
            
            episode_reward += rew
            if dones:
                print("Reward:",episode_reward)
                episode_reward = 0.0
                obs = env.reset()
                break

if __name__ == "__main__":
    main()