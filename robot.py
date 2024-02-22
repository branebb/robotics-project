import numpy as np
from shared import R

def M(l):
    return np.array([
        [1, 0, -l],
        [1, 0, l],
        [0, 1, 0]
    ], dtype=float)

def M_inv(l):
    return np.array([
        [1 / 2, 1 / 2, 0],
        [0, 0, 1],
        [-1 / 2 / l, 1 / 2 / l, 0]
    ], dtype=float)

class Robot:
    __slots__ = ["I_xi", "l", "r", "cpr", "sensor"]
    
    def __init__(self, I_xi=np.zeros(3, dtype=float)):
        self.I_xi = I_xi
        self.l = 0.2
        self.r = 0.05
  
    def update_state(self, I_xi_dot, dt):
        self.I_xi += I_xi_dot * dt
        self.I_xi[2] = np.arctan2(np.sin(self.I_xi[2]), np.cos(self.I_xi[2]))
        
    def update_state_R(self, R_xi_dot, dt):
        I_xi_dot = R(self.I_xi[2]).T @ R_xi_dot
        self.update_state(I_xi_dot, dt)

    def R_inverse_kinematics(self, R_xi_dot):
        return 1 / self.r * M(self.l) @ R_xi_dot

    def forward_kinematics(self, phis_dot):
        return self.r * R(self.I_xi[2]).T @ M_inv(self.l) @ phis_dot
    