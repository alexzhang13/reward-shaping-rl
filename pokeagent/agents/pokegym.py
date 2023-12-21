import numpy as np
from gym.spaces import Box, Space
from itertools import chain

from poke_env import AccountConfiguration
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.data.gen_data import GenData
from poke_env.player import (
    Gen8EnvSinglePlayer,
)
from poke_env.player import RandomPlayer

from pokeagent.agents.max_damage import MaxDamagePlayer


class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=0.0, hp_value=0.0, victory_value=1.0
        )

    def embed_battle_naive(self, battle: AbstractBattle):
        """
        Represent a Pokemon battle (attributes) using a specifsic type of embedding.
        """
        
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    type_1=battle.opponent_active_pokemon.type_1,
                    type_2=battle.opponent_active_pokemon.type_2,
                    type_chart=GenData.from_gen(8).type_chart,
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return battle # np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
    

class PokeGen8Gym(SimpleRLPlayer):
    def __init__(self, set_team=True, opponent="heuristic", *args, **kwargs):
        self.mmap = self.moves_map()
        self.team = self.extract_team()
        self.pkmnmap = self.pkmn_map()
        self.statusmap = self.status_map()  
        self.input_size = 58

        if opponent == "heuristic":
            self.opponent = MaxDamagePlayer(battle_format="gen8ubers", team=self.team)
        else:
            self.opponent = RandomPlayer(battle_format="gen8ubers", team=self.team)

        my_account_config = AccountConfiguration("DeepQLearningAgent", None)
        super(SimpleRLPlayer, self).__init__(team=self.team, opponent=self.opponent, battle_format="gen8ubers", start_challenging=True, *args, **kwargs)
    
    def embed_battle(self, battle: AbstractBattle):
        """
        Represent a Pokemon battle as an embedding. Specified in paper.

        Returns np.float32() embedding, AbstractBattle
        """
        STAT_NORM = 500.0
        STATUS_NORM = 6
        MOVE_NORM = 24
        # active moves
        mvs = [self.mmap[str(move).split(' ')[0]] / MOVE_NORM for move in battle.available_moves]
        for i in range(4 - len(battle.available_moves)):
            mvs.append(0)
        
        # pokemon
        stats = []
        for key, pkmn in battle.team.items():
            pkmn_id = self.pkmnmap[str(pkmn).split(' ')[0]]
            if pkmn.status is not None:
                str_status = self.statusmap[str(pkmn.status).split(' ')[0]]
            else: 
                str_status = 0
            stat = [float(pkmn.active), float(pkmn.fainted), float(str_status / STATUS_NORM), pkmn.current_hp / STAT_NORM, *[val / STAT_NORM for (key,val) in pkmn.stats.items()]]
            stats += stat # [float(pkmn.active), float(pkmn.fainted), float(str_status / STATUS_NORM), float(pkmn.current_hp / STAT_NORM), *[float(val / STAT_NORM) for (key,val) in pkmn.stats.items()]]
        
        emb = mvs + stats
        emb = np.asarray(emb, dtype=np.float32)

        return emb, battle

    def extract_team(self):
        with open('data/team1.txt') as f:
            team = f.read()
        return team

    def moves_map (self):
        with open('data/moves.txt') as f:
            moves = f.readlines()
        counter = 0
        unique_moves = {}
        for move in moves:
            if move not in unique_moves:
                unique_moves[move.replace(" ", "").lower().strip()] = counter
                counter += 1

        return unique_moves
    
    def pkmn_map (self):
        with open('data/pkmn_lists.txt') as f:
            pkmns = f.readlines()
        counter = 0
        unique_pkmn = {}
        for pkmn in pkmns:
            if pkmn not in unique_pkmn:
                unique_pkmn[pkmn.replace(" ", "").lower().strip()] = counter
                counter += 1

        return unique_pkmn
    
    def status_map (self):
        with open('data/status_list.txt') as f:
            pkmns = f.readlines()
        counter = 0
        unique_pkmn = {}
        for pkmn in pkmns:
            if pkmn not in unique_pkmn:
                unique_pkmn[pkmn.strip()] = counter
                counter += 1

        return unique_pkmn