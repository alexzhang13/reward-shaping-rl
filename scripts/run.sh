wandb_gen_id=$(uuidgen)
wandb_name=qlearn
python3 -u -m qlearn \
      +exp=qlearn \
      +wandb_gen_id=$wandb_gen_id \
      hydra.run.dir=outputs/${wandb_name}_$wandb_gen_id \
      wandb_name=${wandb_name} \
      cost_budget=1 \
      train_iterations=100 \
      llm_type=gpt-3.5-turbo

