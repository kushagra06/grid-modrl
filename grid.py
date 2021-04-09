import numpy as np
import gym
from gym import spaces

class GridEnv(gym.Env):
    grid_size = 4
    reward_range = (-1, 0)
    action_space = spaces.Discrete(4)
    observation_space = spaces.Discrete(grid_size * grid_size)
    #left, up, right, down
    action_dict = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}

    def __init__(self, goal=15):
        self.gridworld = np.arange(self.observation_space.n).reshape(4,4)
        # P[a,s,s']
        self.P = np.zeros((self.action_space.n, self.observation_space.n, self.observation_space.n))
        self.P[:, 0, 0] = 1

        for s in self.gridworld.flat[1:-1]:
            r, c = np.argwhere(self.gridworld == s)[0]
            for a, d in self.action_dict.items():
                next_r = max(0, min(r+d[0], 3))
                next_c = max(0, min(c+d[1], 3))
                next_s = self.gridworld[r,c]
                self.P[a,s,next_s] = 1
        
        self.R = np.full((self.action_space.n, self.observation_space.n), -1)
        self.R[:,0] = 0
        self.goal = goal

        self.cur_state = 0

    def get_reward(self, new_state):        
        if new_state == self.goal:
            return 10
        else:
            return 0
    
    def step(self, a):
        prev_state = self.cur_state
        coord = self.state_to_coord(prev_state)

        if a == 0: #left
            if coord[1] == 0:
                r = self.get_reward(prev_state)
            else:
                coord[1] = coord[1] - 1
                self.cur_state = self.coord_to_state(coord)
                r = self.get_reward(self.cur_state)
        
        elif a == 1: #up
            if coord[0] == 0:
                r = self.get_reward(prev_state)
            else:
                coord[0] = coord[0] - 1
                self.cur_state = self.coord_to_state(coord)
                r = self.get_reward(self.cur_state)
        
        elif a == 2: #right
            if coord[1] == self.grid_size - 1:
                r = self.get_reward(prev_state)
            else:
                coord[1] = coord[1] + 1
                self.cur_state = self.coord_to_state(coord)
                r = self.get_reward(self.cur_state)
        
        elif a == 3: #down
            if coord[0] == self.grid_size - 1:
                r = self.get_reward(prev_state)
            else:
                coord[0] = coord[0] + 1
                self.cur_state = self.coord_to_state(coord)
                r = self.get_reward(self.cur_state)
        
        done = 1 if self.cur_state == self.goal else 0
        
        return prev_state, a, self.cur_state, r, done
    
    def state_to_coord(self, state):
        return [state // self.grid_size, state % self.grid_size]
    
    def coord_to_state(self, coord):
        return coord[0] * self.grid_size + coord[1]

    def reset(self):
        self.cur_state = 0
        

# class GridModule(GridEnv):
#     def __init__(self, goal):
#         super().__init__()
#         self.P[:,goal,goal] = 1
#         self.R[:,goal] = 0

# g = GridModule(15)
# print(g.observation_space.n)

        