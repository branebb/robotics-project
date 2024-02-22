import numpy as np
from shared import init_plot_2D, R_array, update_wedge, normalize_angle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import colors
from robot import Robot
from matplotlib.patches import Wedge, Rectangle
from matplotlib.collections import LineCollection


from world import generate_world
from inside import inside_polygon, inside_polygon_robot, expand_points

sign = -1

def path_planning(i):
    v = 0.8
    omega = 3 / 4 * np.pi
    if i % 30 == 0:
        global sign
        sign *= -1
    return np.array([v, 0, sign * omega], dtype=float)

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

    result_obstacles = np.array(result_obstacles)

    result = result_outer_wall | np.any(result_obstacles, axis=0)

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

def animate(i, robot, shapes, dt):
    R_xi_dot = path_planning(i)
    phis_dot = robot.R_inverse_kinematics(R_xi_dot)
    I_xi_dot = robot.forward_kinematics(phis_dot)
    robot.update_state(I_xi_dot, dt)

    update_wedge(shapes[0], robot.I_xi)

if __name__ == "__main__":
    fig, ax = init_plot_2D(lim_from=-3.5, lim_to=3.5)

    num_frames = 100
    fps = 30
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

    for i in range(len(points)):
        rectangle = ax.add_patch(
            Rectangle(
                points[i],
                0.1,
                0.1,
                angle=-90,
                edgecolor=colors.lightgray,
                facecolor=points_color[i],
                zorder=0
            )
        )
        
        rectangles.append(rectangle)


    shapes.append(robot_patch)
    shapes.append(rectangles)

    shapes[0].set_color(colors.blue)
    shapes[0].set_alpha(0.5)

    shapes[1][round((robot.I_xi[0] + 2.95) * 10) + abs(round((robot.I_xi[1] - 2.95) * 600))].set_color(colors.red)
    shapes[1][round((goal[0] + 2.95) * 10) + abs(round((goal[1] - 2.95) * 600))].set_color(colors.green)

    # ani = FuncAnimation(
    #     fig,
    #     animate,
    #     fargs=(robot, shapes, dt),
    #     frames=num_frames,
    #     interval=dt * 1000,
    #     repeat=False,
    #     blit=False,
    #     init_func=lambda: None
    # )
    
    plt.show()