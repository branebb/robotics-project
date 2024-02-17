import numpy as np
from shared import init_plot_2D, R_array, update_wedge, normalize_angle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import colors
from robot import Robot
from matplotlib.patches import Wedge, Circle
from matplotlib.collections import LineCollection

from motion import sample_motion_model_odometry

from sampler import sampler

from world import world
from inside import inside_polygon
from ray_casting import ray_casting

from particles import transform_particles_beams, measurement_model

sign = -1
def path_planning(i):
    v = 0.8
    omega = 3 / 4 * np.pi
    if i % 30 == 0:
        global sign
        sign *= -1
    return np.array([v, 0, sign * omega], dtype=float)


def animate(i, robot, estimated_robot, enc_robot, particles, shapes, dt):
    R_xi_dot = path_planning(i)
    #robot.update_state_R(R_xi_dot, dt)
    phis_dot = robot.R_inverse_kinematics(R_xi_dot)
    I_xi_dot = robot.forward_kinematics(phis_dot)
    robot.update_state(I_xi_dot, dt)


    lines_sensor = robot.measure(lines_env)
    shapes[4].set_segments(lines_sensor)

    deltas_c = enc_robot.read_encoders(phis_dot, dt)
    phis_dot_enc = enc_robot.enc_deltas_to_phis_dot(deltas_c, dt)
    I_xi_dot_enc = enc_robot.forward_kinematics(phis_dot_enc)

    # ovo je crtica x_{t - 1}
    I_xi_prev = enc_robot.I_xi.copy()
    enc_robot.update_state(I_xi_dot_enc, dt)
    # ovo je crtica x_t
    I_xi = enc_robot.I_xi.copy()

    update_wedge(shapes[0], robot.I_xi)
    update_wedge(shapes[3], enc_robot.I_xi)

    # ovo ćemo zamijeniti s vjerojatnosnim modelom gibanja
    #I_xi_dot_particles = np.tensordot(
    #    R_xi_dot, R_array(particles[:, 2]), axes=(0, 1)
    #)
    #particles += I_xi_dot_particles * dt
    #particles += np.random.normal(0, var_movement, (M, 3))

    particles[:] = sample_motion_model_odometry(np.vstack([I_xi_prev, I_xi]), particles)
    particles[:, 2] = normalize_angle(particles[:, 2])
    
    particles_color[:] = colors.green
    inside_obstacle = inside_polygon(particles, lines_env[4:8]) \
        | inside_polygon(particles, lines_env[8:12]) \
        | inside_polygon(particles, lines_env[12:16]) \
        | inside_polygon(particles, lines_env[16:20]) \
        | inside_polygon(particles, lines_env[20:23]) \
        | inside_polygon(particles, lines_env[23:29]) \
        | inside_polygon(particles, lines_env[29:35])

    outside = ~inside_polygon(particles, lines_env[0:4])
    particles_color[inside_obstacle | outside] = colors.red
    
    shapes[2].set_offsets(particles[:, :2])
    shapes[2].set_color(particles_color)

    if i % 5 == 0:
        particles_beams = transform_particles_beams(particles, beams)
        # ovo je za svaku česticu, za svaku zraku z_t^{k*}
        particles_beams = ray_casting(lines_env, particles_beams).reshape((M, lines_sensor.shape[0], 2, 2))

        weights = measurement_model(particles_beams, robot)
        weights[inside_obstacle | outside] = 0
        weights = weights / np.sum(weights)
           
        indices = np.random.choice(np.arange(M), M, p=weights)
        particles[:] = particles[indices]
            
            
    # ovo nismo natipkali na vježbama,
    # ali je preporuka u knjizi Probabilistic robotics
    # Rješenje radi i bez toga, ali s time 
    # dodatno svakih x iteracija, ako je aritmetička sredina svih čestica odaljenija
    # od 0.5m od robota, nasumično odaberemo n čestica te ih uniformno distribuiramo.
    # Time osiguravamo da roj čestica ne iskonvergira u neku točku daleko od robota             
    if i % 10 == 0 and i > 0:
        if np.linalg.norm(np.mean(particles, axis=0)[:2] - robot.I_xi[:2]) >= 0.5:
            n = int(M * 0.75)
            choice = np.random.choice(M, n)
            
            particles[choice, :2] = np.random.uniform(-3, 3, (n, 2))
            particles[choice, 2] = np.random.uniform(-np.pi, np.pi, n)            

    mean = np.mean(particles, axis=0)
    mean[2] = normalize_angle(mean[2])
    estimated_robot.I_xi = mean
    update_wedge(shapes[1], estimated_robot.I_xi)


if __name__ == "__main__":
    fig, ax = init_plot_2D(lim_from=-4.0, lim_to=4.0)

    num_frames = 150
    fps = 30
    dt = 1 / fps

    shapes = []
    
    lines_env = world
    ax.add_collection(
        LineCollection(lines_env, linewidth=1, color=colors.darkgray)
    )
    

    robot = Robot(np.array([0.7, 1.5, np.pi * 4 / 3], dtype=float))
    
    M = 500
    particles = np.random.uniform(-3, 3, (M, 3))
    particles[:, 2] = normalize_angle(particles[:, 2])
    
    particles_color = np.empty(particles.shape[0], dtype=object)
    particles_color[:] = colors.green
    
    inside_obstacle = inside_polygon(particles, lines_env[4:8]) \
        | inside_polygon(particles, lines_env[8:12]) \
        | inside_polygon(particles, lines_env[12:16]) \
        | inside_polygon(particles, lines_env[16:20]) \
        | inside_polygon(particles, lines_env[20:23]) \
        | inside_polygon(particles, lines_env[23:29]) \
        | inside_polygon(particles, lines_env[29:35])
    
    outside = ~inside_polygon(particles, lines_env[0:4])
    
    particles_color[inside_obstacle | outside] = colors.red

    particles_shape = ax.scatter(
         particles[:, 0], particles[:, 1], s=0.5, color=particles_color
    )

    mean = np.mean(particles, axis=0)
    mean[2] = normalize_angle(mean[2])

    estimated_robot = Robot(mean)

    enc_robot = Robot(robot.I_xi.copy())

    robot_patch = ax.add_patch(
        Wedge(
            robot.I_xi[:2],
            0.2,
            np.rad2deg(robot.I_xi[2]) + 10,
            np.rad2deg(robot.I_xi[2]) - 10,
            zorder=2
        )
    )

    estimated_robot_patch = ax.add_patch(
        Wedge(
            estimated_robot.I_xi[:2],
            0.2,
            np.rad2deg(estimated_robot.I_xi[2]) + 10,
            np.rad2deg(estimated_robot.I_xi[2]) - 10,
            facecolor=colors.yellow, alpha=0.7, zorder=3
        )
    )

    enc_robot_patch = ax.add_patch(
        Wedge(
            enc_robot.I_xi[:2],
            0.2,
            np.rad2deg(enc_robot.I_xi[2]) + 10,
            np.rad2deg(enc_robot.I_xi[2]) - 10,
            facecolor=colors.darkgray, alpha=0.7, zorder=3
        )
    )

    shapes.append(robot_patch)
    shapes.append(estimated_robot_patch)
    shapes.append(particles_shape)
    shapes.append(enc_robot_patch)

    lines_sensor = robot.measure(lines_env)
    lines_sensor_collection = ax.add_collection(
        LineCollection(lines_sensor, linewidth=1, color=colors.cyan, zorder=1)
    )
    shapes.append(lines_sensor_collection)
    
    beams = np.expand_dims(robot.sensor.S_beams, axis=0).repeat(M, axis=0) 
    

    particles_beams = transform_particles_beams(particles, beams)
    # ovo je za svaku česticu, za svaku zraku z_t^{k*}
    particles_beams = ray_casting(lines_env, particles_beams).reshape((M, lines_sensor.shape[0], 2, 2))

    """
    shapes.append(
        ax.add_collection(
            LineCollection(particles_beams[5], color=colors.orange)
        )
    )"""

    ani = FuncAnimation(
        fig,
        animate,
        fargs=(robot, estimated_robot, enc_robot, particles, shapes, dt),
        frames=num_frames,
        interval=dt * 1000,
        repeat=False,
        blit=False,
        init_func=lambda: None
    )
    
    
    plt.show()