# Simple Reward Shaping with LLM Feedback for RL
This repository is designed to be a minimalist framework for using LLMs on a generic RL (gym) environment. In this example, we use classic gym environments like MountainCarv0 and Pokemon Showdown (poke-env).

## Setup
Set up the Pokemon Showdown server first:
```
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```
To set up [poke_env](https://github.com/hsahovic/poke-env) and the related libraries:
``` 
conda create -n "reward_shaping" python=3.8
pip install -e .
```


## The General Method
The general concept behind reward shaping is to introduce an extra intrinsic reward, independent of the environment API, to the model. This reward is used in the computation of value functions, and therefore gradient updates in the case of deep models, and is generally a function of (state, action, next state).

### Reward Shaping + Reward Shaping w/ LLMs

### Models
1. Generic tabular Q-learning.
2. Generic DQN + w/ Modifications.
3. Generic PPO + w/ Modifications.

