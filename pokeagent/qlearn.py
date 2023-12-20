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
    DISCRETE_OBSERVATION_SPACE_SIZE = [20] * len(
        env.observation_space.high
    )  # will give out 20*20 list

    # see how big is the range for each of the 20 different buckets
    discrete_os_win_size = (
        env.observation_space.high - env.observation_space.low
    ) / DISCRETE_OBSERVATION_SPACE_SIZE

    LEARNING_RATE = cfg.lr
    DISCOUNT = 0.95  # how important we find the new future actions are ; future reward over current reward
    EPISODES = cfg.train_iterations
    render = False

    epsilon = (
        0.5  # 0-1 ; higher it is, more likely for it to perform something random action
    )
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES // 2
    epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

    q_table = np.random.uniform(
        low=-2, high=0, size=(DISCRETE_OBSERVATION_SPACE_SIZE + [env.action_space.n])
    )

    def get_discrete_state(state):
        discrete_state = (state - env.observation_space.low) / discrete_os_win_size
        return tuple(discrete_state.astype(np.int32))  # return as tuple

    rewards = []
    lengths = []
    sr = ShapedReward()

    for ep in range(EPISODES):
        done = False
        discrete_state = get_discrete_state(env.reset()[0])  # initial discrete state

        if ep % 500 == 0:
            render = True
        else:
            render = False
            env.close()

        episode_length = 0
        sum_rewards = 0
        shaped_reward = 0
        shaped_reward_func = sr.generate_default_func()
        complete_goal = 0
        traj = []

        while not done and episode_length < 600:  # goal reached means reward = 0
            if np.random.random() > epsilon:
                # in this environment, 0 means push the car left, 1 means to do nothing, 2 means to push it right
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Run one timestep of the environment's dynamics;  returns a tuple (observation, reward, done, info).
            
            new_state, reward, done, info, _ = env.step(action)
            new_discrete_state = get_discrete_state(new_state)

            traj.append((discrete_state, action, reward))
            shaped_reward = shaped_reward_func(new_discrete_state[0], new_discrete_state[1], action)

            if render:
                env.render()

            if not done:
                # max q value for the next state calculated above
                max_future_q = np.max(q_table[new_discrete_state])

                # q value for the current action and state
                current_q = q_table[discrete_state + (action,)]

                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                    reward + shaped_reward + DISCOUNT * max_future_q
                )

                # based on the new q, we update the current Q value
                q_table[discrete_state + (action,)] = new_q

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
        if END_EPSILON_DECAYING >= ep >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

        wandb.log({"train": {"rewards": sum_rewards, "episode_length": episode_length}})
        print(sr.dump())

    env.close()


if __name__ == "__main__":
    main()
