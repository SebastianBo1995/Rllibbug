import os
import gym
from gym import spaces
import numpy as np
import ray

class dummyenv(gym.Env):
    """Umgebung zur Simulation in GT
    Als Variablen die vom Agenten vreädnert werden können zählt die Drosselklappenstellung und das Wastegate
    """
    metadata = {'render.modes': ['console']} # console
    def __init__(self, env_config):
        super(dummyenv, self).__init__()
        self.action_space = spaces.Box(low=np.array([0.1, 0]), high=np.array([1, 1]), dtype=np.float32) # Drosselklappe, Wastegate
        self.observation_space = spaces.Box(low=0, high=255, shape=(2,), dtype=np.float32)
        observation = [0,0]

    def step(self, action):
        reward = 0
        info = {}
        observation= np.array([1000, 10], dtype=np.float32)
        done = True
        return observation, reward, done , info

    def reset(self):
        observation = np.array([4000, 10], dtype=np.float32)
        return observation  # reward, done, info can't be included

    def close(self):
          pass


trainer = 'SAC'
# Setzen der allgemeinen Randbedingungen:
config = {}

# [Hyperparameter]
config['num_workers'] = 1 # Anzahl der Arbeiter, für jeden Arbeiter wird eine Umgebung und bei GT-Rechnung eine Lizenz belegt!

# [Modell-Einstellungen]
if trainer == 'SAC':
    config['Q_model'] = {'fcnet_hiddens': [100, 100]} # für SAC
    config['policy_model'] = {'fcnet_hiddens': [100, 100]} # für SAC
    config['learning_starts'] = 1
    config['timesteps_per_iteration'] = 1
    config['target_network_update_freq'] = 1
    #config['n_step']=1

if trainer == 'PPO':
    config['model'] = {'fcnet_hiddens': [100, 100]} # für PPO
    config['sgd_minibatch_size'] = 1

# [Evaluierung]
config['evaluation_interval'] = 24
config['evaluation_num_episodes'] = 10
config['evaluation_config'] = {'explore': False}
config['evaluation_num_workers'] = 0 # 0 = Berechnung auf worker für training

# [Fortgeschrittene Einstellungen]
config['num_gpus'] = 0 # Anzahl der GPUs bei der Berechnung, 0 = keine GPU-Berechnung
config['framework'] = 'tf2' # tf, tfe, tf2 verfügbar
config['eager_tracing'] = True

config['rollout_fragment_length'] = 1
config['train_batch_size'] = 1
config['explore'] = False
config['normalize_actions'] = True

# [Verwenden vorheriger Daten]
config['input'] = os.path.join(os.path.dirname(__file__), 'Output_Data_{}'.format(trainer))
config['input_evaluation'] = []

ray.init()

if trainer == 'PPO':
    from ray.rllib.agents import ppo
    agent = ppo.PPOTrainer(env=dummyenv, config=config)
    
elif trainer == 'SAC':
    from ray.rllib.agents import sac
    agent = sac.SACTrainer(env=dummyenv, config=config)

for n in range(10):
    result = agent.train()