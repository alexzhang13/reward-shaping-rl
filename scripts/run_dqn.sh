wandb_gen_id=$(uuidgen)
wandb_name=dqn_poke
python3 -u pokeagent/main.py \
      +exp=dqn_poke \
      wandb_mode=disabled \
      +wandb_gen_id=$wandb_gen_id \
      hydra.run.dir=outputs/debug-${wandb_gen_id} \
      cost_budget=1 \
      train_iterations=500 \
      max_episodes=4000 \
      llm_type=gpt-3.5-turbo

