import numpy as np
from shared import init_plot_2D, R_array, update_wedge, normalize_angle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import colors
from robot import Robot
from matplotlib.patches import Wedge, Rectangle
from matplotlib.collections import LineCollection, PatchCollection

from world import generate_world
from inside import inside_polygon, inside_polygon_robot, expand_points

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

def in_near_obstacle(expanded_points, lines_env):

    outer_wall_region = lines_env[0:4]
    obstacles = [lines_env[4:7], lines_env[7:11], lines_env[11:17], lines_env[17:20],
               lines_env[20:24], lines_env[24:30], lines_env[30:33], lines_env[33:37],
               lines_env[37:43]]

    inside_outer_wall = np.array([~inside_polygon(point, outer_wall_region) for point in expanded_points])

    result_outer_wall = np.any(inside_outer_wall, axis=1)

    result_obstacles = []

    for obstacle in obstacles:
        inside_obstacles = np.any(np.array([inside_polygon(point, obstacle) for point in expanded_points]), axis=1)
        result_obstacles.append(inside_obstacles)

    result_obstacles = np.any(np.array(result_obstacles), axis=0)

    result = result_outer_wall | result_obstacles
    return result

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
        
def heuristic(x, y, goal):
    return abs(goal[0] - x) + abs(goal[1] - y)

def find_path(robot_position, goal_position, grid):

    goal = [round(abs((goal_position[1] - 2.95) * 10)),
            round((goal_position[0] + 2.95) * 10)]
    
    delta = [[-1, 0],
             [0, -1],
             [1, 0],
             [0, 1],
             [1, 1],
             [1, -1],
             [-1, 1],
             [-1, -1]]
    cost = 1
    
    closed = [[0 for row in range(60)] for col in range(60)]
    closed[round(abs((robot_position[1] - 2.95) * 10))][round((robot_position[0] + 2.95) * 10)] = 1

    expand = [[-1 for row in range(60)] for col in range(60)]
    parent = [[[-1, -1] for row in range(60)] for col in range(60)]


    x = round(abs((robot_position[1] - 2.95) * 10))
    y = round((robot_position[0] + 2.95) * 10)
    g = 0
    h = heuristic(x, y, goal_position)
    f = g + h

    open = [[f, g, h, x, y]]

    found = False
    resign = False
    count = 0

    while found is False and resign is False:
        if len(open) == 0:
            resign = True
            print("Fail")

        else:
            open.sort()
            open.reverse()
            next = open.pop()
            g = next[1]
            x = next[3]
            y = next[4]
            expand[x][y] = count
            count += 1

            if x == goal[0] and y == goal[1]:
                found = True
            
            else:
                for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < 60 and y2 >= 0 and y2 < 60:
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            h2 = heuristic(x2, y2, goal)
                            f2 = g2 + h2
                            open.append([f2, g2, h2, x2, y2])
                            closed[x2][y2] = 1
                            parent[x2][y2] = [x, y]

    path = []
    x, y = goal
    while y != round((robot_position[0] + 2.95) * 10) or x != round(abs((robot_position[1] - 2.95) * 10)):
        path.append([x, y])
        x, y = parent[x][y]
    path.append([round(abs((robot_position[1] - 2.95) * 10)), round((robot_position[0] + 2.95) * 10)])
    path.reverse()
    
    return path

def animate(i, robot, shapes, path_points, dt):
    global continue_animation
    
    if len(path_points):

        if np.linalg.norm(robot.I_xi[:2] - path_points[0]) < 0.05:
            path_points.pop(0)

        if len(path_points):
            target_point = path_points[0]

            v, omega = path_planning(robot.I_xi, target_point)

            phis_dot = robot.R_inverse_kinematics(np.array([v, 0, omega], dtype=float))
            I_ksi_dot = robot.forward_kinematics(phis_dot)
            robot.update_state(I_ksi_dot, dt)

            update_wedge(shapes[0], robot.I_xi)
    else:
        continue_animation = False

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

def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

if __name__ == "__main__":
    fig, ax = init_plot_2D(lim_from=-3.5, lim_to=3.5)

    max_frames = 1000
    fps = 15
    dt = 1 / fps

    shapes = []

    xspace = np.arange(-3, 3, 0.1, dtype=float)
    yspace = np.arange(3, -3, -0.1, dtype=float)
    X, Y = np.meshgrid(xspace, yspace)
    points = np.dstack([X, Y])
    points = points.reshape(-1, 2)

    points_color = np.empty(points.shape[0], dtype=object)
    points_color[:] = colors.white

    lines_env = generate_world()

    ax.add_collection(
        LineCollection(lines_env, linewidth=1, color=colors.darkgray)
    )
    
    robot = generate_robot(lines_env)

    points_mean = points + np.array([0.05, -0.05])

    expaneded_points = expand_points(points_mean)
    
    in_near = in_near_obstacle(expaneded_points, lines_env)

    points_color[in_near] = colors.lightgray

    goal = generate_goal(robot, lines_env)

    robot_patch = ax.add_patch(
        Wedge(
            robot.I_xi[:2],
            0.2,
            np.rad2deg(robot.I_xi[2]) + 10,
            np.rad2deg(robot.I_xi[2]) - 10,
            zorder=2
        )
    )

    rectangles = []
    for i, near_obstacle in enumerate(in_near):
            rectangle = Rectangle(
                    points[i],
                    0.1,
                    0.1,
                    facecolor=points_color[i],
                    edgecolor=colors.lightgray,
                    angle=-90,
                    zorder=0,
                )
            rectangles.append(rectangle)

    # Thanks to Ramal Salha for his contribution! 
    ax.add_collection(PatchCollection(rectangles,match_original=True))
    
    shapes.append(robot_patch)
    shapes[0].set_color(colors.blue)
    shapes[0].set_alpha(0.5)

    grid = np.array(in_near.astype(int), dtype=int).reshape((60, 60))

    ideal_path = find_path(robot.I_xi[:2], goal, grid)
    ideal_path = ideal_path[1:-1]
    path_points = []

    rectangles2 = []

    for i in range(len(ideal_path)):
        path_points.append([round(ideal_path[i][1] * 0.1 - 3, 2) + 0.05, round(-1 * ideal_path[i][0] * 0.1 + 3, 2) - 0.05])
        rectangle = Rectangle(
                [round(ideal_path[i][1] * 0.1 - 3, 2), round(-1 * ideal_path[i][0] * 0.1 + 3, 2)],
                0.1,
                0.1,
                angle=-90,
                zorder=0
            )
        rectangles2.append(rectangle)
    ax.add_collection(PatchCollection(rectangles2, color=colors.yellow))
   
    start = Rectangle(
                    [robot.I_xi[0] - 0.05, robot.I_xi[1] + 0.05],
                    0.1,
                    0.1,
                    angle=-90,
                    color=colors.red,
                    zorder=0
                )
    finish = Rectangle(
                    [goal[0] - 0.05, goal[1] + 0.05],
                    0.1,
                    0.1,
                    angle=-90,
                    color=colors.green,
                    zorder=0
                )
    ax.add_collection(PatchCollection([start, finish], match_original=True))
    
    first_point = path_points[0]
    target_angle = np.arctan2(first_point[1] - robot.I_xi[1], first_point[0] - robot.I_xi[0])
    robot.I_xi[2] = target_angle

    update_wedge(shapes[0], robot.I_xi)

    path_points.append([goal[0], goal[1]])

    ani = FuncAnimation(
        fig,
        animate,
        fargs=(robot, shapes, path_points, dt),
        frames=max_frames,
        interval=1,
        repeat=True,
        blit=False,
        init_func=lambda: None
    )
    # writer = animation.PillowWriter(fps=30,
    #                                 metadata=dict(artist='Branimir Brekalo'),
    #                                 bitrate=1800)
    # ani.save('A_star2.gif', writer=writer)
    plt.show()
    