import numpy as np
from shared import init_plot_2D, R_array, update_wedge, normalize_angle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import colors
from robot import Robot
from matplotlib.patches import Wedge, Rectangle
from matplotlib.collections import LineCollection


from world import generate_world
from inside import inside_polygon, inside_polygon_robot

def generate_robot(lines_env):

    while True:
        robot = Robot(np.array([np.random.uniform(-2.6, 2.6),
                            np.random.uniform(-2.6, 2.6),
                            np.random.uniform(0, 2 * np.pi)], dtype=float))
        
        robot_position = np.array([robot.I_xi], dtype=float)

        inside_obstacle = inside_polygon_robot(robot_position, lines_env[4:7], 0.4) \
                        | inside_polygon_robot(robot_position, lines_env[7:11], 0.4) \
                        | inside_polygon_robot(robot_position, lines_env[11:17], 0.4) \
                        | inside_polygon_robot(robot_position, lines_env[17:20], 0.4) \
                        | inside_polygon_robot(robot_position, lines_env[20:24], 0.4) \
                        | inside_polygon_robot(robot_position, lines_env[24:30], 0.4) \
                        | inside_polygon_robot(robot_position, lines_env[30:33], 0.4) \
                        | inside_polygon_robot(robot_position, lines_env[33:37], 0.4) \
                        | inside_polygon_robot(robot_position, lines_env[37:43], 0.4)
    
        if not inside_obstacle:
            return robot

def animate(i, robot, shapes, dt):
    # print(i)

    # if i < len(shapes[1]):
    #     shapes[1][i].set_facecolor(colors.red)
    
    # return shapes
    return 0

if __name__ == "__main__":
    fig, ax = init_plot_2D(lim_from=-3.5, lim_to=3.5)

    num_frames = 150
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

    
    inside_obstacle = inside_polygon(points_mean, lines_env[4:7]) \
                        | inside_polygon(points_mean, lines_env[7:11]) \
                        | inside_polygon(points_mean, lines_env[11:17]) \
                        | inside_polygon(points_mean, lines_env[17:20]) \
                        | inside_polygon(points_mean, lines_env[20:24]) \
                        | inside_polygon(points_mean, lines_env[24:30]) \
                        | inside_polygon(points_mean, lines_env[30:33]) \
                        | inside_polygon(points_mean, lines_env[33:37]) \
                        | inside_polygon(points_mean, lines_env[37:43]) \

    points_color[inside_obstacle] = colors.lightgray

    robot_patch = ax.add_patch(
        Wedge(
            robot.I_xi[:2],
            0.2,
            np.rad2deg(robot.I_xi[2]) + 10,
            np.rad2deg(robot.I_xi[2]) - 10,
            zorder=2
        )
    )

    robot_position = np.array([robot.I_xi], dtype=float)

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

    shapes[0].set_color(colors.red)
    
    
    # ani = FuncAnimation(
    #     fig,
    #     animate,
    #     fargs=(robot, shapes, dt),
    #     frames=num_frames,
    #     interval=dt * 1,
    #     repeat=False,
    #     blit=False,
    #     init_func=lambda: None
    # )
    
    
    plt.show()