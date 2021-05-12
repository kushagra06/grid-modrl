import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
Transition_done = namedtuple('Transition_done', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    def __init__(self, capacity=256):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size=8):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class ReplayMemory2:
    def __init__(self, capacity=256):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition_done(*args))
    
    def sample(self, batch_size=8):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)