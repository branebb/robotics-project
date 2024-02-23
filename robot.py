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
    __slots__ = ["I_xi", "l", "r"]
    
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

class PID:
    __slots__ = ["k_p", "k_i", "k_d", "e_prev", "e_acc", "dt"]

    def __init__(self, dt, k_p=3.0, k_i=1.5, k_d=0.2):
        self.dt = dt
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.e_prev = self.e_acc = 0

    def reset(self):
        self.e_prev = self.e_acc = 0

    def __call__(self, e):
        e_diff = (e - self.e_prev) / self.dt
        self.e_acc += e * self.dt
        self.e_prev = e
        return self.k_p * e + self.k_i * self.e_acc + self.k_d * e_diff
class GoToGoalAlgorithm:
    __slots__ = ["pid", "robot", "I_g", "dt"]

    def __init__(self, robot, I_g, dt):
        self.pid = PID(dt)
        self.robot = robot
        self.I_g = I_g
        self.dt = dt

    def __call__(self):
        e = np.arctan2(
            self.I_g[1] - self.robot.I_xi[1],
            self.I_g[0] - self.robot.I_xi[0],
        ) - self.robot.I_xi[2]
        e = np.arctan2(np.sin(e), np.cos(e))

        R_v = self.robot.r * (100 * 2 * np.pi / 60)
        R_omega = self.pid(e)

        phis_dot = self.robot.R_inverse_kinematics(
            np.array([R_v, 0, abs(R_omega)], dtype=float)
        )
        phi_dot_lrmax = max(phis_dot[0], phis_dot[1])
        phi_dot_lrmin = min(phis_dot[0], phis_dot[1])

        if phi_dot_lrmax > (100 * 2 * np.pi / 60):
            phi_dot_ld = phis_dot[0] - (phi_dot_lrmax - (100 * 2 * np.pi / 60))
            phi_dot_rd = phis_dot[1] - (phi_dot_lrmax - (100 * 2 * np.pi / 60))
        elif phi_dot_lrmin < 0:
            phi_dot_ld = phis_dot[0] + (0 - phi_dot_lrmin)
            phi_dot_rd = phis_dot[1] + (0 - phi_dot_lrmin)
        else:
            phi_dot_ld = phis_dot[0]
            phi_dot_rd = phis_dot[1]

        phis_dot_ld = max(0, min((100 * 2 * np.pi / 60), phi_dot_ld))
        phis_dot_rd = max(0, min((100 * 2 * np.pi / 60), phi_dot_rd))

        R_ksi_dot = R(self.robot.I_xi[2]) @ self.robot.forward_kinematics(
            np.array([phis_dot_ld, phis_dot_rd, 0], dtype=float)
        )
        R_v_feas = R_ksi_dot[0]
        R_omega_feas = np.copysign(1, R_omega) * R_ksi_dot[2]

        return R_v_feas, R_omega_feas
    
class StopAlgorithm:
    def __call__(self):
        return 0, 0

class PathPlanning:
    __slots__= ["robot", "I_g", "dt", "go_to_goal", "stop", "d_1", "algorithm"]    
    
    def __init__(self, robot, I_g, dt):
        self.robot = robot
        self.I_g = I_g
        self.dt = dt
        self.go_to_goal = GoToGoalAlgorithm(robot, I_g, dt) 
        self.stop = StopAlgorithm()
        self.d_1 = 0.05
        self.algorithm = self.go_to_goal

    def __call__(self):
        
        current_dist_from_goal = np.linalg.norm(self.robot.I_xi[:2] - self.I_g[:2])

        at_goal = current_dist_from_goal < self.d_1
        
        if at_goal:
            # print("At goal")
            self.algorithm = self.stop
        elif not at_goal:
            self.algorithm = self.go_to_goal
        else:
            0
            
        return self.algorithm()    