#imports
import numpy as np
import matplotlib.pyplot as plt
import colors

from astar import find_path
from world import generate_world
from shared import init_plot_2D, update_wedge
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge, Rectangle
from inside import expand_points, in_near_obstacle
from robot import generate_robot, generate_goal, path_planning
from matplotlib.collections import LineCollection, PatchCollection

#animation
def animate(i, robot, shapes, path_points, dt):
    #do animation if there is some points on path not visited yet
    if len(path_points):

        #if robot is in the 0.05 or less radius remove point from the list
        if np.linalg.norm(robot.I_xi[:2] - path_points[0]) < 0.05:
            path_points.pop(0)

        #if there is some point left
        #using path planning calculate linear velocity and desired angle
        #update robot
        if len(path_points):
            target_point = path_points[0]

            v, omega = path_planning(robot.I_xi, target_point)

            phis_dot = robot.R_inverse_kinematics(np.array([v, 0, omega], dtype=float))
            I_ksi_dot = robot.forward_kinematics(phis_dot)
            robot.update_state(I_ksi_dot, dt)

            update_wedge(shapes[0], robot.I_xi)

if __name__ == "__main__":
    #initial settings
    fig, ax = init_plot_2D(lim_from=-3.5, lim_to=3.5)
    max_frames = 150
    fps = 15
    dt = 1 / fps
    shapes = []

    #dividing space from -3 to 3 ito parts with length 0.1 (both axis)
    #creating meshgrid with all coordinates
    xspace = np.arange(-3, 3, 0.1, dtype=float)
    yspace = np.arange(3, -3, -0.1, dtype=float)
    X, Y = np.meshgrid(xspace, yspace)
    points = np.dstack([X, Y])
    points = points.reshape(-1, 2)

    #initial color for all points (each point is just up left corner of square)
    points_color = np.empty(points.shape[0], dtype=object)
    points_color[:] = colors.white

    #generating world and adding it to collection
    lines_env = generate_world()

    ax.add_collection(
        LineCollection(lines_env, linewidth=1, color=colors.darkgray)
    )
    
    #generating robot
    robot = generate_robot(lines_env)

    #shifting original points from meshgrid 
    #reason: this points are exactly in the middle of squares
    points_mean = points + np.array([0.05, -0.05])
    
    #expanding each point with another x amount (currently 135)
    #for checking if the point is in or near some obstacle
    expaneded_points = expand_points(points_mean)
    
    #generating True/False array for all points (3600)
    #near or in obstacle True, otherwise False
    in_near = in_near_obstacle(expaneded_points, lines_env)

    #setting only near or in obstacle points in gray color
    points_color[in_near] = colors.lightgray

    #generating goal
    goal = generate_goal(robot, lines_env)

    #separate patch for robot
    robot_patch = ax.add_patch(
        Wedge(
            robot.I_xi[:2],
            0.2,
            np.rad2deg(robot.I_xi[2]) + 10,
            np.rad2deg(robot.I_xi[2]) - 10,
            zorder=2
        )
    )

    #creating rectangles with white/gray color and saving them into rectangles array
    #adding them into collection
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
    
    #adding robot to shapes for updates in animations
    #coloring and opacity added
    shapes.append(robot_patch)
    shapes[0].set_color(colors.blue)
    shapes[0].set_alpha(0.5)

    #transforming True/False array into 60x60 0/1 grid for path algorithm
    grid = np.array(in_near.astype(int), dtype=int).reshape((60, 60))

    #calculating ideal path
    #removing first and last point on path (coloring reasons)
    ideal_path = find_path(robot.I_xi[:2], goal, grid)
    ideal_path = ideal_path[1:-1]
    
    #creating path and rectangles array
    #adding path points (centered)
    #creating rectangles from ideal path
    #adding them to collection
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
   
    #adding separate rectangles for start and finish squares
    #adding them to collection later
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
    
    #adjusting robots angle based on the first point in the path
    #reason: so he just starts going that way and doesn't rotate and make big circle
    first_point = path_points[0]
    target_angle = np.arctan2(first_point[1] - robot.I_xi[1], first_point[0] - robot.I_xi[0])
    robot.I_xi[2] = target_angle
    update_wedge(shapes[0], robot.I_xi)

    #adding goal as last point into path, not added initially because of the coloring
    path_points.append([goal[0], goal[1]])

    #final animation settings
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

    # settings for saving into gif
    # writer = animation.PillowWriter(fps=30,
    #                                 metadata=dict(artist='Branimir Brekalo'),
    #                                 bitrate=1800)
    # ani.save('A_star.gif', writer=writer)

    plt.show()
    