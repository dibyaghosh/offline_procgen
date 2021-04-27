#########
# NOTE:
#   I used this wrapper to actually collect the procgen datasets. To do so, 
#   I added the following line after L38 of train.py in the train-procgen repo
#   https://github.com/openai/train-procgen
#
#    venv = VecOfflineLogger(venv, filename=f'{log_dir}/buffer')
###########


import numpy as np
from baselines.common.vec_env import VecEnvWrapper
import gzip
import pickle
class VecOfflineLogger(VecEnvWrapper):
    def __init__(self, venv, filename=None, bufsize=int(5e6)):
        super().__init__(venv)
        inner_env = venv
        while hasattr(inner_env, 'venv'):
            inner_env = inner_env.venv
        self.inner_env = inner_env.env
        self.transitions = []
        self.filename = filename
        self.buffer = dict()
        self.infos = dict()
        self.verify_images = np.zeros((bufsize // 2500, 64, 64, 64, 3))
        self.env_states = dict()
        self.cursor = 0
        self.bufsize = bufsize
        self._last_saved = 0
    
    def reset(self):
        return self.venv.reset()

    def add_transition(self, actions, rewards, dones, infos):
        if self.cursor >= self.bufsize:
            return

        if 'actions' not in self.buffer:
            self.buffer['actions'] = np.zeros((self.bufsize, *actions.shape), dtype=actions.dtype)
            self.buffer['rewards'] = np.zeros((self.bufsize, *rewards.shape), dtype=rewards.dtype)
            self.buffer['dones'] = np.zeros((self.bufsize, *dones.shape), dtype=dones.dtype)
        self.buffer['actions'][self.cursor] = actions
        self.buffer['rewards'][self.cursor] = rewards
        self.buffer['dones'][self.cursor] = dones

        for i in range(len(infos)):
            for k in infos[i]:
                if k not in self.infos and np.array(infos[i][k]).dtype != object:
                    arr = np.array(infos[i][k])
                    self.infos[k] = np.zeros((self.bufsize, len(infos), *arr.shape), dtype=arr.dtype)
                self.infos[k][self.cursor, i] = infos[i][k]
        self.cursor = self.cursor + 1
        if self.cursor % 500 == 0:
            print(self.cursor)
    def save_buffer(self, filename=None):
        if not filename:
            filename = self.filename
        if filename:
            print(f'Saving buffer to {filename}_mainbuffer.gz, with verification at {filename}_verification.npz')
            main_dict = dict(states=self.env_states,
                             **{k: v[:self.cursor] for k, v in self.buffer.items()}
                            )
            with gzip.open(f'{filename}_mainbuffer.gz', 'wb') as f:
                pickle.dump(main_dict, f)            
            np.savez_compressed(f'{filename}_verification.npz',
                verify_images=self.verify_images[:(self.cursor // 2500) + 1],
                **{k: v[:self.cursor] for k, v in self.buffer.items()},
                **{k: v[:self.cursor] for k, v in self.infos.items()},
            )
            
    
    def step_async(self, actions):
        if self.cursor % 2500 == 0:
            states = self.inner_env.callmethod("get_state")
            self.env_states[self.cursor] = states
        
        self.actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        assert self.actions is not None
        obs, rews, dones, infos = self.venv.step_wait()
        if self.cursor % 2500 == 0:
            self.verify_images[self.cursor // 2500] = obs.copy()

        self.add_transition(self.actions, rews, dones, infos)
        self.actions = None
        if self.cursor - self._last_saved > 10000:
            self.save_buffer()
            self._last_saved = self.cursor
        return obs, rews, dones, infos