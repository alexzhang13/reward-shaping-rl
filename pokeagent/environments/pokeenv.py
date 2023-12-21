import asyncio

import numpy as np
from gym.spaces import Box, Space
import logging
from gym.utils.env_checker import check_env

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.data.gen_data import GenData
from poke_env.player import (
    Gen8EnvSinglePlayer,
    RandomPlayer,
)

from pokeagent.agents.pokegym import PokeGen8Gym
from pokeagent.models.dqn import DQNAgent

def train(env: PokeGen8Gym, agent: DQNAgent, episodes:int):
    for ep in range(episodes):
        print('-=-=-=-=- NEW EP:', ep)
        state, info = env.reset()
        s, battle = state
        steps = 0
        average_loss = 0
        while True:
            
            # agent step and learn
            action = agent.action(s) # [agent.action(state)]
            new_state, reward, terminated, truncated, info = env.step(action)
            new_s, new_battle = new_state[0], new_state[1]
            done = terminated or truncated
            agent.cache(s, action, reward, new_s, done)
            q, loss = agent.optimize()
            # logger.log_step(reward, loss, q)
            
            # state = new_state
            s = new_s
            
            if done:
                print('done!', done)
                break
            
            if loss > 0:
                average_loss += loss
                steps += 1
                
        # log episode info
        # logger.log_episode()
        if ep > 0 and ep % 500 == 0:
            evaluate(agent, 20)

        average_loss = loss / steps if (steps > 0) else -1
        logging.info('average_loss', average_loss)
        
    env.close()

def evaluate(agent: DQNAgent, episodes:int):
    eval_env = PokeGen8Gym(set_team=True, opponent="random") # change later

    for ep in range(episodes):
        state, info = eval_env.reset()
        s, battle = state
        while True:
            
            # agent step and learn
            action = agent.action(s) # [agent.action(state)]
            new_state, reward, terminated, truncated, info = eval_env.step(action)
            new_s, new_battle = new_state[0], new_state[1]
            done = terminated or truncated
            
            # state = new_state
            s = new_s
            
            if done:
                logging.info(f'eval step {ep}/{episodes}')
                print('done!', done)
                break
                
    logging.info(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    eval_env.close()
    

async def main():
    # First test the environment to ensure the class is consistent
    # with the OpenAI API
    test_env = PokeGen8Gym(set_team=True, opponent="random")
    # check_env(test_env)
    test_env.close()

    # Create one environment for training and one for evaluation
    # opponent = RandomPlayer(battle_format="gen8randombattle")
    train_env = PokeGen8Gym(set_team=True, opponent="random")
    # opponent = RandomPlayer(battle_format="gen8randombattle")
    # eval_env = SimpleRLPlayer(
        # battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    # )

    # Compute dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape
    
    n_steps = 10
    done = False
    while not done:
        # Random action
        action = train_env.action_space.sample()
        (obs, battle), reward, done, info, what = train_env.step(action)
        print(battle, obs, reward, done)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
