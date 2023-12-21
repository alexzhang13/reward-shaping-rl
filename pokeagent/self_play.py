import asyncio
from pokeagent.environments import pokeenv
from pokeagent.models.dqn import (
    DQNAgent
)
from pokeagent.agents.pokegym import PokeGen8Gym
from poke_env.player import RandomPlayer
from pokeagent.agents.max_damage import MaxDamagePlayer

async def battle_handler(player1, player2, num_challenges):
    await asyncio.gather(
        player1.agent.accept_challenges(player2.username, num_challenges),
        player2.agent.send_challenges(player1.username, num_challenges),
    )

# def training_function(player, model, model_kwargs):
#     # Fit (train) model as necessary.
#     model.fit(player, **model_kwargs)
#     player.done_training = True
#     # Play out the remaining battles so both fit() functions complete
#     # We use 99 to give the agent an invalid option so it's forced
#     # to take a random legal action
#     while player.current_battle and not player.current_battle.finished:
#         _ = player.step(99)

# if __name__ == "__main__":
#     ...
#     player1 = SimpleRLPlayer(
#         battle_format="gen8randombattle",
#         log_level=30,
#         opponent="placeholder",
#         start_challenging=False,
#     )
#     player2 = SimpleRLPlayer(
#         battle_format="gen8randombattle",
#         log_level=30,
#         opponent="placeholder",
#         start_challenging=False,
#     )
#     ...
#     # Self-Play bits
#     player1.done_training = False
#     player2.done_training = False
#     # Get event loop
#     loop = asyncio.get_event_loop()
#     # Make two threads: one per player and each runs model.fit()
#     t1 = Thread(target=lambda: train_m1(player1, ppo, p1_env_kwargs))
#     t1.start()

#     t2 = Thread(target=lambda: training_function(player2, ppo, p2_env_kwargs))
#     t2.start()
#     # On the network side, keep sending & accepting battles
#     while not player1.done_training or not player2.done_training:
#         loop.run_until_complete(battle_handler(player1, player2, 1))
#     # Wait for thread completion
#     t1.join()
#     t2.join()

    player1.close(purge=False)
    player2.close(purge=False)