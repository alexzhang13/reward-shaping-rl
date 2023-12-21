import numpy as np


class ReplayMemory:
    def __init__(self, max_capacity=10000):
        self._capacity = 0
        self._max_capacity = max_capacity
        self._idx = 0

        self._memory = []

    def sample(self, batch_size):
        random_indices = np.random.randint(0, self._capacity - 1, batch_size)
        states, actions, rewards, states_prime, dones = None, None, None, None, None

        for idx in random_indices:
            s, a, r, s_prime, done = self._memory[idx]
            # s = np.transpose(s, (0, 3, 1, 2))
            # s_prime = np.transpose(s_prime, (0, 3, 1, 2))

            states = (
                np.expand_dims(s, axis=0) 
                if states is None 
                else np.concatenate((states, np.expand_dims(s, 0)))
            )
            states_prime = (
                np.expand_dims(s_prime, 0)
                if states_prime is None
                else np.concatenate((states_prime, np.expand_dims(s_prime, 0)))
            )
            rewards = (
                np.expand_dims(r, 0)
                if rewards is None
                else np.concatenate((rewards, np.expand_dims(r, 0)))
            )
            actions = (
                np.expand_dims(a, 0)
                if actions is None
                else np.concatenate((actions, np.expand_dims(a, 0)))
            )
            dones = (
                np.expand_dims(done, 0)
                if dones is None
                else np.concatenate((dones, np.expand_dims(done, 0)))
            )
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(states_prime),
            np.array(dones),
        )

    def store(self, s, a, r, sprime, done):
        """Store observation into replay memory

        Args:
            s (ndarray): state
            a (int): action
            r (float): reward
            sprime (ndarray): next state
            done (bool): if state is last state
        """
        data = (s, a, r, sprime, done)

        if self._capacity >= self._max_capacity:
            self._memory[self._idx] = data
        else:
            self._memory.append(data)
            self._capacity += 1
        self._idx = (self._idx + 1) % self._max_capacity
