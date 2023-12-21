import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pokeagent.models.memory import ReplayMemory
from pokeagent.models.agent import Agent


class DQN(nn.Module):
    """
    Base DQN class for running RL models.
    Args:
        embedding_size: size of input embedding
    """

    def __init__(self, embedding_size, num_actions=4):
        super().__init__()
        self._embedding_size = embedding_size
        self._num_actions = num_actions

        # mario: (4, 84, 84)
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self._num_actions),
            # nn.Conv2d(self._num_frames, out_channels=32, kernel_size=8, stride=4),
            # nn.ReLU(),
            # nn.Conv2d(32, out_channels=64, kernel_size=4, stride=2),
            # nn.ReLU(),
            # nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1),
            # nn.ReLU(),
            # nn.Flatten(),
            # nn.Linear(64 * 7 * 7, 512),
            # nn.ReLU(),
            # nn.Linear(512, self._num_actions),
        )

    def forward(self, obs):
        actions = self.net(obs)
        return actions


class DQNAgent(Agent):
    """
    Agent using DQN (Double DQN) implementation to make choices.
    """

    def __init__(
        self,
        embedding_size,
        num_actions,
        device,
        evaluate,
        batch_size=16,
        gamma=0.9,
        lr=1e-4,
        epsilon_decay=0.99999975,
        epsilon_min=0.1,
        save_model_iter=10000,
        sync_model_iter=10000,
        update_freq=5,
        warmup=10000,
        name='dqn',
        save_dir="../output/",
    ):
        self._embedding_size = embedding_size
        self._num_actions = num_actions
        self._gamma = gamma
        self._batch_size = batch_size
        self._replay_memory = ReplayMemory()
        self._epsilon = 1.0
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._target_network = DQN(embedding_size, num_actions)
        self._policy_network = DQN(embedding_size, num_actions)
        self._target_network.load_state_dict(self._policy_network.state_dict())
        self._curr_step = 0
        self._save_model_iter = save_model_iter
        self._sync_model_iter = sync_model_iter
        self._warmup = warmup
        self._save_dir = save_dir
        self._update_freq = update_freq
        self._device = device
        self._evaluate = evaluate
        self._name = name

        self._target_network = self._target_network.to(device)
        self._policy_network = self._policy_network.to(device)
        self._optimizer = torch.optim.Adam(self._policy_network.parameters(), lr=lr)

    def cache(self, state, action, reward, next_state, done):
        self._replay_memory.store(state, action, reward, next_state, done)

    def td_estimate(self, state, action):
        q_values = self._policy_network(state)
        q_values = q_values.gather(1, action.unsqueeze(1)).squeeze()
        return q_values

    def td_target(self, reward, next_state, done):
        with torch.no_grad():
            target_next_q = self._policy_network(next_state)
            best_action = torch.argmax(target_next_q, axis=1)
            q_values = target_next_q.gather(1, best_action.unsqueeze(1)).squeeze()

            return reward.squeeze() + self._gamma * (1 - done.squeeze()) * q_values

    def sync_target(self):
        """
        Sync weights between current Q network and target
        """
        self._target_network.load_state_dict(self._policy_network.state_dict())

    def optimize(self):
        """
        Optimize TD error/loss from DQN agent
        Returns V(s) and R
        """

        # sync or save model
        if self._curr_step < self._warmup:
            return None, None
        if self._curr_step % self._update_freq:
            return None, None

        if self._curr_step % self._save_model_iter == 0:
            self.save()
        if self._curr_step % self._sync_model_iter == 0:
            self.sync_target()

        # batch of samples from experience
        s, a, r, sprime, done = self._replay_memory.sample(self._batch_size)

        # normalize pixels and send to gpu
        s = np.array(s) / 255.0
        sprime = np.array(sprime) / 255.0
        s = torch.from_numpy(s).float().to(self._device)
        sprime = torch.from_numpy(sprime).float().to(self._device)
        r = torch.from_numpy(r).float().to(self._device)
        a = torch.from_numpy(a).to(self._device)
        done = torch.from_numpy(done).int().to(self._device)

        # compute td targets and estimate for loss
        td_estimate = self.td_estimate(s, a)
        td_target = self.td_target(r, sprime, done)

        if td_target.shape != td_estimate.shape:
            pass

        # compute loss and backpropogate
        loss = F.smooth_l1_loss(td_estimate, td_target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        del s, sprime

        return (td_estimate.mean().item(), loss.item())

    def action(self, state):
        """
        Select epsilon-greedy action to take
        """
        with torch.no_grad():
            choose = np.random.uniform(0, 1)
            if choose < self._epsilon and not self._evaluate:
                action = np.random.randint(0, self._num_actions)
            else:
                # state = np.transpose(state, (0, 3, 1, 2))
                state = torch.from_numpy(state).float().to(self._device)
                state /= 255.0
                q_values = self._policy_network(state)
                # print()
                action = torch.argmax(q_values, axis=0).item()

            self._epsilon *= self._epsilon_decay
            self._epsilon = max(self._epsilon_min, self._epsilon)
            self._curr_step += 1

            return action

    def save(self):
        save_path = (
            self._save_dir
            / f"{self._name}_dqn_net_{int(self._curr_step // self._save_model_iter)}.chkpt"
        )
        torch.save(
            dict(
                model=self._policy_network.state_dict(), exploration_rate=self._epsilon
            ),
            save_path,
        )
        print(f"DQN saved to {save_path} at step {self._curr_step}")
    
    def save_all(self):
        save_path = (
            self._save_dir
            / f"{self._name}_dqn_net_final.chkpt"
        )
        torch.save(
            dict(
                model=self._policy_network.state_dict(), exploration_rate=self._epsilon
            ),
            save_path,
        )
        print(f"DQN saved to {save_path} after training finished")

    def load(self, path, device):
        self._policy_network.load_state_dict(
            torch.load(path, map_location=device)["model"]
        )
        print(f"DQN loaded from to {path} at step {self._curr_step}")
