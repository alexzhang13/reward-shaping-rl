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
from pokeagent.models.dqn import PPO1
from pokeagent.utils.reward import ShapedReward
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO1

"""
Adapted for stable baselines PPO. I really need to clean this up later...
"""

def train_m1(env: PokeGen8Gym, agent: PPO1, episodes:int, sr:ShapedReward=None, device=None, save_dir=None):
    """
    Training method 1: Sequential learning of reward function. Code is a bit jank but it gets the job done for now.
    """
    META_STEPS = 5
    shaped_reward = 0
    shaped_reward_func = sr.generate_default_func()
    
    for meta_step in range(META_STEPS):
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
                shaped_reward = shaped_reward_func(battle, new_battle)

                agent.cache(s, action, reward + shaped_reward, new_s, done)
                q, loss = agent.optimize()
                # logger.log_step(reward, loss, q)
                
                # state = new_state
                s = new_s
                battle = new_battle
                
                if done:
                    print('done!', done)
                    break
                
                if loss is not None and loss > 0:
                    average_loss += loss
                    steps += 1
                    
            # log episode info
            # logger.log_episode()
            if ep > 0 and ep % 500 == 0:
                evaluate(agent, 20)

            if (steps > 0):
                average_loss = average_loss / steps 
            else:
                average_loss = -1
                
            logging.info('average_loss', average_loss)
        agent.save_all()
        won, total_games = evaluate(agent, 20)
        shaped_reward_func = sr.generate_reward_func(won / total_games)
        agent = PPO1(MlpPolicy, env, verbose=1)
        sr.save()
    env.close()
    sr.save()

def train_m2(env: PokeGen8Gym, agent: PPO1, episodes:int, sr:ShapedReward=None, device=None, save_dir=None):
    """
    Training method 2: Tree-based. Takes a while because I'm not using threading or async training...
    """
    META_STEPS = 5
    NUM_LEAVES = 5
    shaped_reward = 0
    shaped_reward_func = sr.generate_default_func()
    
    for meta_step in range(META_STEPS):
        MAX_REWARDS = []
        for k in range(NUM_LEAVES):
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
                    shaped_reward = shaped_reward_func(battle, new_battle)

                    agent.cache(s, action, reward + shaped_reward, new_s, done)
                    q, loss = agent.optimize()
                    # logger.log_step(reward, loss, q)
                    
                    # state = new_state
                    s = new_s
                    battle = new_battle
                    
                    if done:
                        print('done!', done)
                        break
                    
                    if loss is not None and loss > 0:
                        average_loss += loss
                        steps += 1
                        
                # log episode info
                # logger.log_episode()
                if ep > 0 and ep % 500 == 0:
                    evaluate(agent, 20)

                if (steps > 0):
                    average_loss = average_loss / steps 
                else:
                    average_loss = -1
                    
                logging.info('average_loss', average_loss)
            agent.save_all()
            won, total_games = evaluate(agent, 20)
            shaped_reward_func = sr.generate_reward_func(won / total_games)
            agent = PPO1(embedding_size=env.input_size, 
                    num_actions=env.action_space.n,
                    device=device,
                    evaluate=False,
                    lr=0.001,
                    save_dir=save_dir,
                    warmup=100,
                    name="iterate_{meta_step}_{k}")
            MAX_REWARDS.append(won / total_games)
        sr.save()
    env.close()
    sr.save()

def train_m3(env: PokeGen8Gym, agent: PPO1, episodes:int, sr:ShapedReward=None, device=None, save_dir=None):
    """
    Training method 3 
    """
    shaped_reward = 0
    shaped_reward_func = sr.generate_default_func() # sr.generate_reward_func([])
    sr.save()
    
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
            shaped_reward = shaped_reward_func(battle, new_battle)

            agent.cache(s, action, reward + shaped_reward, new_s, done)
            q, loss = agent.optimize()
            # logger.log_step(reward, loss, q)
            
            # state = new_state
            s = new_s
            battle = new_battle
            
            if done:
                print('done!', done)
                break
            
            if loss is not None and loss > 0:
                average_loss += loss
                steps += 1
                
        # log episode info
        # logger.log_episode()
        if ep > 0 and ep % 500 == 0:
            won, total_games = evaluate(agent, 20)
            # shaped_reward_func = sr.generate_reward_func(won / total_games)
            sr.save()

        if (steps > 0):
            average_loss = average_loss / steps 
        else:
            average_loss = -1
            
        logging.info('average_loss', average_loss)
        
    env.close()
    agent.save_all()
    sr.save()

def evaluate(agent: PPO1, episodes:int):
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
        f"PPO Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    eval_env.close()
    return eval_env.n_won_battles, eval_env.n_finished_battles
