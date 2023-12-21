from poke_env.player import RandomPlayer, background_cross_evaluate
from pokeagent.environments import pokeenv
from pokeagent.models.dqn import (
    DQNAgent
)
from pokeagent.agents.pokegym import PokeGen8Gym
from pokeagent.environments.pokeenv import evalw

from tabulate import tabulate
import torch

def main():
    env = PokeGen8Gym(set_team=True, opponent="heuristic", log_level=25)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using CUDA: {use_cuda}")
    model = DQNAgent(embedding_size=env.input_size, 
                num_actions=env.action_space.n,
                device=device,
                evaluate=False,
                lr=0.001,
                warmup=100)
    path = 'output/2023-12-21T12-44-53/dqn_dqn_net_final.chkpt'
    model.load(path=path, device=device)
    
    # Create players to eval
    # players = [RandomPlayer(max_concurrent_battles=10) for _ in range(3)]
    players = [
        env.agent,
        RandomPlayer(battle_format="gen8randombattle"),
    ]

    # Cross evaluate players: each player plays 20 games against every other player
    cross_evaluation = background_cross_evaluate(players, n_challenges=20)
    evalw(DQNAgent, env, episodes=20)

    # Prepare results for display
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])

    # Display results
    print(tabulate(table))

if __name__ == "__main__":
    main()
    