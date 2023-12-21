import gym
import yaml
import numpy as np
import random
import datetime
import torch
from pathlib import Path
from distutils.util import strtobool
import torch.multiprocessing as _mp
import os
import sys

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import numpy as np
import gym
import logging

from utils.setup import (
    setup_wandb,
    set_seeds,
)

# A logger for this file
log = logging.getLogger(__name__)

from pokeagent.utils.reward import ShapedReward
from pokeagent.agents.pokegym import PokeGen8Gym
from pokeagent.environments import pokeenv
from pokeagent.models.dqn import (
    DQNAgent
)

def train(cfg, env, device, mp=None):
    # set up logging and saving
    save_dir = Path("output/") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    
    # TODO: Change model config selection to be more general
    model = DQNAgent(embedding_size=env.input_size, 
                num_actions=env.action_space.n,
                device=device,
                evaluate=False,
                lr=0.001,
                save_dir=save_dir,
                warmup=100)
    pokeenv.train(env=env, 
                agent=model, 
                episodes=cfg.max_episodes)
    
    # elif config['model']['model_name'] == 'PPO':
    #     model = PPO.PPOAgent(obs_shape=env.num_states,
    #                         num_actions=env.num_actions,
    #                         lr=config['model']['learning_rate'],
    #                         num_envs=config['environment']['num_envs'],
    #                         num_rollout_steps=config['model']['rollout_steps'],
    #                         gamma=config['model']['gamma'],
    #                         gae_lambda=config['model']['gae_lambda'],
    #                         clip_coeff = config['model']['clip_coeff'],
    #                         entropy_coeff = config['model']['entropy_coeff'],
    #                         value_loss_coeff = config['model']['value_loss_coeff'],
    #                         max_grad_norm = config['model']['max_grad_norm'],
    #                         target_kl = config['model']['target_kl'],
    #                         device=device)
    
    # if config['environment']['sync_vector_env']:
    #     ENV.train_vectorized(env,
    #                         agent=model,
    #                         writer=writer,
    #                         mp=mp,
    #                         config=config,
    #                         num_envs=config['environment']['num_envs'],
    #                         batch_size=config['model']['batch_size'],
    #                         num_steps=config['environment']['max_step'],
    #                         device=device,
    #                         render=False)
    # else:
        

def evaluate(config, env, device):
    print('Evaluate.')
    '''
    if config['model']['model_name'] == 'DQN':
            model = DQN.DQNAgent(num_frames=config['environment']['n_stack'], 
                                num_actions=env.action_space.n,
                                device=device,
                                evaluate=True,
                                lr=config['model']['learning_rate'],
                                warmup=0)
    elif config['model']['model_name'] == 'PPO':
        model = PPO.PPOAgent(obs_shape=np.array(env.num_states),
                                num_actions=env.num_actions,
                                lr=config['model']['learning_rate'],
                                num_envs=config['environment']['num_envs'],
                                num_rollout_steps=config['model']['rollout_steps'],
                                gamma=config['model']['gamma'],
                                gae_lambda=config['model']['gae_lambda'],
                                clip_coeff = config['model']['clip_coeff'],
                                device=device)
    model.load(args.model_weights_path, device)
    ENV.evaluate(env=env, agent=model, render=True)
    '''


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Log the config to terminal
    print(OmegaConf.to_yaml(cfg))

    logging.basicConfig(
        # handlers=[
        #     logging.StreamHandler(sys.stdout),
        #     logging.FileHandler("output.log", mode="w"),
        # ],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    print(f"Logging to file {os.path.join(hydra_cfg['run']['dir'], 'eval.log')}")

    # Set seeds
    set_seeds(cfg.seed)

    # Setup wandb
    wandb_conf = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    setup_wandb(wandb_conf)
    
    # Set up vectorized (multiple) or single environment
    # if config['environment']['sync_vector_env']:
    #     # open multi-processing
    #     mp = _mp.get_context("spawn")
    #     env = MultipleEnvironments(config['environment']['world'], 
    #                                config['environment']['stage'], 
    #                                config['environment']['action_type'], 
    #                                config['environment']['num_envs'])
        
    #     print(dir(env.envs[0]))
    #     assert isinstance(env.single_action_space, gym.spaces.Discrete) # assert gym env
        
    #     print('Action Space:', env.single_action_space)
    #     print('Observation Space:', env.single_observation_space.shape)
    #     print('Config Model Keys', config['model'].keys())
    #     # env = gym.vector.SyncVectorEnv([ENV.env_init(stacked_frames=False, seed=config['SEED'] + i) \
    #     #                                 for i in range(config['environment']['num_envs'])])
    # else:
    env = PokeGen8Gym(set_team=True, opponent="heuristic", log_level=25)
    # ENV.env_init(n_stack=config['environment']['n_stack'], seed=config['SEED'])()
    
    # check CUDA config
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using CUDA: {use_cuda}")
    print()
    
    if cfg.evaluate:
        evaluate(cfg, env, device)
    else:
        train(cfg, env, device, ) # mp=mp)
        

if __name__ == "__main__":
    main()
