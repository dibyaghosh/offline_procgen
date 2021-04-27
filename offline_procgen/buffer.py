import numpy as np
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
import gzip
import pickle

class OfflineProcgenReplayBuffer:
    def __init__(self, env_name, buffer_fname,
                    num_levels=200, start_level=0, distribution_mode='easy', ordering_rng=None):
        self.venv_parameters = dict(num_envs=64, env_name=env_name, num_levels=num_levels, start_level=0, distribution_mode=distribution_mode, rand_seed=0)
        print('Creating inner Procgen env with parameters: ', self.venv_parameters)
        venv = ProcgenEnv(**self.venv_parameters)
        venv = VecExtractDictObs(venv, "rgb")
        self.venv = venv

        print('Loading buffer')
        with gzip.open(buffer_fname, 'rb') as f:
            self.buffer = pickle.load(f)
        self.n_chunks = len(self.buffer['states']) - 1
        print(f'There are {self.n_chunks} sets of 2500 transitions in the buffer. Loading 1 at a time.')
        if not ordering_rng:
            ordering_rng = np.random.default_rng()
        self.orderings = np.array(
            [ordering_rng.permutation(self.n_chunks) for _ in range(64)]
        )

        self._observations = np.zeros((64, 2501, 64, 64, 3), dtype=np.uint8)
        self._actions = np.zeros((64, 2500), dtype=int)
        self._rewards = np.zeros((64, 2500))
        self._terminals = np.zeros((64, 2500), dtype=bool)
        self.load_chunk(0)

    @property
    def loaded_buffer(self):
        return dict(observations=self._observations[:, :-1],
                    next_observations=self._observations[:, 1:], 
                    actions=self._actions,
                    rewards=self._rewards,
                    terminals=self._terminals)

    def load_chunk(self, idx=None):
        if idx is None:
            idx = np.random.choice(self.n_chunks)
        starting_indices = [
            2500 * ordering[idx]
            for ordering in self.orderings
        ]
        print('Starting from timesteps: ', starting_indices)
        start_state = [
            self.buffer['states'][si][k]
            for k, si in enumerate(starting_indices)
        ]
        venv = self.venv
        self.venv.venv.env.callmethod("set_state", start_state)
        obs = venv.venv.env.observe()[1]['rgb']
        for i in range(2500):
            actions = np.array([self.buffer['actions'][i+si][k] for k, si in enumerate(starting_indices)]) 
            next_obs, r, done, info = venv.step(actions)
            ### VERIFICATION
            pred_dones = np.array([self.buffer['dones'][i+si][k] for k, si in enumerate(starting_indices)]) 
            pred_rews = np.array([self.buffer['rewards'][i+si][k] for k, si in enumerate(starting_indices)]) 
            assert all(pred_dones == done), f'Failed on Env {np.argmin(pred_dones == done)} at index {i+starting_indices[np.argmin(pred_dones == done)]}'
            assert all(pred_rews == r), f'Failed on Env {np.argmin(pred_rews == r)} at index {i+starting_indices[np.argmin(pred_rews == r)]}'
            self._observations[:, i] = obs
            self._actions[:, i] = actions
            self._rewards[:, i] = r
            self._terminals[:, i] = done
            obs = next_obs
        self._observations[:, -1] = obs # Final observation
        print(f'Loaded chunk {idx} into memory: {2500 * 64} datapoints')


