wandb_gen_id=$(uuidgen)
now=$(date +"%m_%d_%Y_%M:%S")
wandb_name=dqn_poke
python3 -u pokeagent/main.py \
      +exp=dqn_poke \
      wandb_mode=disabled \
      +wandb_gen_id=$wandb_gen_id \
      hydra.run.dir=outputs/${wandb_name}_${now} \
      cost_budget=1 \
      train_iterations=1 \
      max_episodes=10000 \
      llm_type=gpt-3.5-turbo \
      model=PPO

