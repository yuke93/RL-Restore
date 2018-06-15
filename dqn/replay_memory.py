import random
import numpy as np


class ReplayMemory:
    def __init__(self, config):
        self.memory_size = config.memory_size
        self.actions = np.empty(self.memory_size, dtype = np.uint8)
        self.rewards = np.empty(self.memory_size, dtype = np.float16)
        self.screens = np.empty((self.memory_size, config.screen_height, config.screen_width, config.screen_channel), dtype = np.float16)
        self.terminals = np.empty(self.memory_size, dtype = np.bool)
        self.history_length = 1
        self.dims = (config.screen_height, config.screen_width, config.screen_channel)
        self.batch_size = config.batch_size
        self.count = 0
        self.current = 0
        self.stop_step = config.stop_step
        self.safe_length = self.stop_step + 1

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
        self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)


    def add(self, screen, reward, action, terminal):
        screen_temp = screen.reshape(screen.shape[1:])
        assert screen_temp.shape == self.dims
        # NB! screen is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current, ...] = screen_temp
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size


    def getEpiBatch(self, batch_size):
        s_t = []
        action = []
        reward = []
        # episode may have different lengths
        for _ in range(self.stop_step):
            s_t.append([])
            action.append([])
            reward.append([])
        for _ in range(batch_size):
            cur_episode = self.getEpisode()
            len_cur = len(cur_episode)  # length of current episode
            s_t_cur = s_t[len_cur - 1]
            action_cur = action[len_cur - 1]
            reward_cur = reward[len_cur - 1]
            for m in range(len_cur):
                s_t_cur.append(cur_episode[m, 0])
                action_cur.append(cur_episode[m, 1])
                reward_cur.append(cur_episode[m, 2])
        for k in range(self.stop_step):
            if len(reward[k]) > 0:
                s_t[k] = np.concatenate(s_t[k], axis=0).astype(np.float)
            action[k] = np.array(action[k], dtype=np.int)
            reward[k] = np.array(reward[k], dtype=np.float)

        return s_t, action, reward


    def getEpisode(self):  # return single episode
        assert self.count > self.history_length
        while True:
            index = random.randint(self.history_length + self.safe_length, self.count - 1)
            # if wraps over current pointer, then get new one
            if index - self.history_length - self.safe_length <= self.current <= \
                  index + self.history_length + self.safe_length:
                continue
            # if wraps over episode end, then get new one
            if self.terminals[(index - self.history_length):index].any():
                continue
            # in case touch the end
            if index + self.history_length + self.safe_length >= self.memory_size or \
                  index - self.history_length - self.safe_length <= 0:
                continue

            # search for the start state
            idx_start = index
            while not self.terminals[idx_start - 2]:
                idx_start -= 1
            # search for the end state
            idx_end = index
            while not self.terminals[idx_end]:
                idx_end += 1

            # get the whole episode
            output = []
            for k in range(idx_start, idx_end + 1):
                s_t = self.getState(k - 1).copy()
                action = self.actions[k]
                reward = self.rewards[k]
                s_t_plus_1 = self.getState(k).copy()
                terminals = self.terminals[k]
                output.append([s_t, action, reward, s_t_plus_1, terminals])
            output = np.array(output)
            assert output[-1, -1]
            return output


    def getState(self, index):
        assert self.count > 0
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.history_length - 1:
            # use faster slicing
            return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return self.screens[indexes, ...]

