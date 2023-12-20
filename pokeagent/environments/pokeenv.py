import asyncio

import numpy as np
from gym.spaces import Box, Space
from gym.utils.env_checker import check_env

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.data.gen_data import GenData
from poke_env.player import (
    Gen8EnvSinglePlayer,
    RandomPlayer,
)

from pokeagent.agents.SimpleRLAgent import SimpleRLPlayer


async def main():
    # First test the environment to ensure the class is consistent
    # with the OpenAI API
    opponent = RandomPlayer(battle_format="gen8randombattle")
    test_env = SimpleRLPlayer(
        battle_format="gen8randombattle", start_challenging=True, opponent=opponent
    )
    check_env(test_env)
    test_env.close()

    # Create one environment for training and one for evaluation
    opponent = RandomPlayer(battle_format="gen8randombattle")
    train_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )

    # opponent = RandomPlayer(battle_format="gen8randombattle")
    # eval_env = SimpleRLPlayer(
        # battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    # )

    # Compute dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape
    
    n_steps = 10
    for _ in range(n_steps):
        # Random action
        action = train_env.action_space.sample()
        obs, reward, done, info, what = train_env.step(action)
        print(obs, reward, done)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
