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
We use a really simple approach to reward shaping, which is not complicated but also can be a bit faulty in the current setup. We basically prompt the OpenAI APIs to generate a shaped reward function with the appropriate parameters, then call `exec(code, globals())` to interpret the definition in code. It's a bit jank, but the best reward we've gotten is
```
def reward(prev_battle_state, next_battle_state):
    prev_fainted = [mon for mon in prev_battle_state.opponent_team.values() if mon.fainted]
    next_fainted = [mon for mon in next_battle_state.opponent_team.values() if mon.fainted]
    
    prev_total_hp = sum([mon.current_hp for mon in prev_battle_state.opponent_team.values()])
    next_total_hp = sum([mon.current_hp for mon in next_battle_state.opponent_team.values()])
    
    reward_defeat = len(next_fainted) - len(prev_fainted)
    reward_damage = (prev_total_hp - next_total_hp) / 1000.0
    
    return reward_defeat + reward_damage
```
although this result is with a lot of extra prompting on top. Ideally, we'd like to make this framework a lot less brittle, but given our time and compute constraints, this is what we had to make do with.

### Models
For models, we use tabular Q-learning (for MountainCarv0) example, the [stable_baselines](https://stable-baselines.readthedocs.io/en/master/modules/ppo1.html) implementation of PPO (a bit faulty, I think this wasn't done fully right) and our own implementation of DQN/double DQN. 

### Training and Evaluation

Unfortunately, I didn't have enough time to make this all generalizable and nice since experiments were rushed. So for the time being, you have to fiddle with parameters and code to run the right models. But generally, the right training scripts are
```
bash scripts/run_{model}_m{k}.sh
```
where model is either {dqn,ppo} and k is in {1,2,3}.

And the two relevant evaluation scripts are
```
python cross_evaluate.py
```
```
python self_play.py
```
