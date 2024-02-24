import numpy as np
from shared import R
from inside import inside_polygon_robot

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

#generating robot with random position and orientation
#checking if robot is in/near obstacle and making sure he isnt
def generate_robot(lines_env):
    while True:
        robot = Robot(np.array([np.random.randint(-27, 27) * 0.1 + 0.05,
                            np.random.randint(-27, 27) * 0.1 - 0.05,
                            np.random.randint(0, 2 * np.pi)], dtype=float))
        
        robot_position = np.array([robot.I_xi], dtype=float)

        inside_obstacle = inside_polygon_robot(robot_position, lines_env[4:7], 0.3) \
                        | inside_polygon_robot(robot_position, lines_env[7:11], 0.3) \
                        | inside_polygon_robot(robot_position, lines_env[11:17], 0.3) \
                        | inside_polygon_robot(robot_position, lines_env[17:20], 0.3) \
                        | inside_polygon_robot(robot_position, lines_env[20:24], 0.3) \
                        | inside_polygon_robot(robot_position, lines_env[24:30], 0.3) \
                        | inside_polygon_robot(robot_position, lines_env[30:33], 0.3) \
                        | inside_polygon_robot(robot_position, lines_env[33:37], 0.3) \
                        | inside_polygon_robot(robot_position, lines_env[37:43], 0.3)
    
        if not inside_obstacle:
            return robot

#generating goal with random position
#checking if goal is in/near obstacle and that goal isn't at same position as robot
def generate_goal(robot, lines_env):
    while True:
        goal_position = np.array([np.random.randint(-27, 27) * 0.1 + 0.05,
                                   np.random.randint(-27, 27) * 0.1 - 0.05], dtype=float)
        
        goal_check = np.array([goal_position], dtype=float)

        inside_obstacle = inside_polygon_robot(goal_check, lines_env[4:7], 0.3) \
                        | inside_polygon_robot(goal_check, lines_env[7:11], 0.3) \
                        | inside_polygon_robot(goal_check, lines_env[11:17], 0.3) \
                        | inside_polygon_robot(goal_check, lines_env[17:20], 0.3) \
                        | inside_polygon_robot(goal_check, lines_env[20:24], 0.3) \
                        | inside_polygon_robot(goal_check, lines_env[24:30], 0.3) \
                        | inside_polygon_robot(goal_check, lines_env[30:33], 0.3) \
                        | inside_polygon_robot(goal_check, lines_env[33:37], 0.3) \
                        | inside_polygon_robot(goal_check, lines_env[37:43], 0.3)
    
        if not inside_obstacle and not all(np.equal(goal_position, robot.I_xi[:2])):
            return goal_position

#path planing for current robot position and given point
#returning linear velocity and desired angle 
def path_planning(robot_position, target_point):
    
    target_angle = np.arctan2(target_point[1] - robot_position[1], target_point[0] - robot_position[0])
    desired_angle = normalize_angle(target_angle - robot_position[2])
    distance_to_target = np.linalg.norm(target_point - robot_position[:2])
    
    max_angular_velocity = np.pi / 2

    if np.abs(desired_angle) > np.pi / 4:
        omega = np.clip(np.sign(desired_angle) * max_angular_velocity, -max_angular_velocity, max_angular_velocity)
    else:
        omega = desired_angle  
        
    max_linear_velocity = 1

    v = min(max_linear_velocity, distance_to_target)
    
    return v, omega

# [-pi, pi]
def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))