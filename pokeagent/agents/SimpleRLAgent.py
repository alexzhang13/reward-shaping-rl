import numpy as np
from gym.spaces import Box, Space

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.data.gen_data import GenData
from poke_env.player import (
    Gen8EnvSinglePlayer,
)

class SimpleRLPlayer(Gen7EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=0.0, hp_value=0.0, victory_value=1.0
        )

    def embed_battle(self, battle: AbstractBattle):
        """
        Represent a Pokemon battle as an embedding. Specified in paper.

        Returns np.float32() embedding, AbstractBattle
        """

        return battle

    def embed_battle_naive(self, battle: AbstractBattle):
        """
        Represent a Pokemon battle (attributes) using a specific type of embedding.
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
