import math
import numpy as np

class Reward:
    def __init__(self, finger_com, target_com, qpos_d, hinge2_com):
        self.finger_com = finger_com
        self.target_com = target_com
        self.qpos_d = qpos_d
        self.hinge2_y = hinge2_com[1]
        
        self.finger_y = self.finger_com[0]
        self.target_y = self.target_com[0]
        self._bool_init = [True,True,True]

        self._y_error = 0
        
        # Reward Weight
        self.Rd = 1
        self.Ro = 0.3
        self.Ry = 0.15
        
        self.vec = self.finger_com - self.target_com
        self.reward_dist = np.linalg.norm(self.finger_com - self.target_com)
        
        self.reward_y_er = (self.finger_y - self.target_y)**2
        self.hinge_er = (self.hinge2_y - self.target_y)**2
        
    
    @property
    def reward(self):
        t_reward = - self.Rd * self.reward_dist #- self.Ry * self.reward_y_error  #self.Ro * abs(self.reward_ori)
        return  t_reward
        
    @property
    def is_reach(self):
        if self.reward_dist < 0.05:
            return True
