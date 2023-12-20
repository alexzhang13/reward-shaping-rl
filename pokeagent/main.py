import os

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
from pokeagent.utils.reward import ShapedReward

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Log the config to terminal
    print(OmegaConf.to_yaml(cfg))

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.WARN,
    )

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    print(f"Logging to file {os.path.join(hydra_cfg['run']['dir'], 'eval.log')}")

    # Set seeds
    set_seeds(cfg.seed)

    # Setup wandb
    wandb_conf = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    setup_wandb(wandb_conf)

    env = gym.make("MountainCar-v0")
    done = False
    env.reset()

    LEARNING_RATE = cfg.lr
    DISCOUNT = 0.95  # how important we find the new future actions are ; future reward over current reward
    EPISODES = cfg.train_iterations
    render = False

    rewards = []
    lengths = []
    sr = ShapedReward()

    for ep in range(EPISODES):
        done = False
        discrete_state = get_discrete_state(env.reset()[0])  # initial discrete state

        episode_length = 0
        sum_rewards = 0
        shaped_reward = 0
        shaped_reward_func = sr.generate_default_func()
        complete_goal = 0
        traj = []

        while not done and episode_length < 600:  # goal reached means reward = 0
            new_state, reward, done, info, _ = env.step(action)
            traj.append((discrete_state, action, reward))
            shaped_reward = shaped_reward_func(new_discrete_state[0], new_discrete_state[1], action)

            if render:
                env.render()

            if not done:
                # q update?

            elif new_state[0] >= env.goal_position:
                print(("Goal reached at {} episode".format(ep)))
                q_table[discrete_state + (action,)] = 1
                complete_goal = 1

            discrete_state = new_discrete_state
            episode_length += 1
            sum_rewards += reward

        if complete_goal == 0 and ep < 5:
            shaped_reward_func = sr.generate_reward_func(traj)  

        rewards.append(sum_rewards)
        lengths.append(episode_length)

        wandb.log({"train": {"rewards": sum_rewards, "episode_length": episode_length}})
        print(sr.dump())

    env.close()


if __name__ == "__main__":
    main()
