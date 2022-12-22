import gym
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import frankaEnv

# Parallel environments

    
policy_kwargs = dict(activation_fn=th.nn.Tanh,
                     net_arch=[128, dict(pi=[64, 64], vf=[64, 64])])
    
def main():
    # Parallel environments
    env = frankaEnv.FrankaEnv()

    n_step = 15000 #1000
    total_timestep = 15000 * 200


    model = PPO("MlpPolicy", env, n_steps=n_step, policy_kwargs=policy_kwargs, verbose=1)
    
    model.learn(total_timesteps=total_timestep)
    
    model.save("ppo_franka")

    del model # remove to demonstrate saving and loading

    model = PPO.load("ppo_franka")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    
    
    
if __name__ == "__main__":
  main()