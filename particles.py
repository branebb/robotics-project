import numpy as np
from shared import R_array
from robot import Robot

def transform_particles_beams(particles, beams):
    particles_beams = beams.copy()

    rot = R_array(particles[:, 2])[:, :2, :2]
    transl = np.expand_dims(particles[:, :2], axis=1).repeat(beams.shape[1], axis=1)

    # rotacija
    particles_beams[:, :, 1, :] = np.matmul(particles_beams[:, :, 1, :], rot)

    # transl. po x
    particles_beams[:, :, 0, 0] += transl[:, :, 0]
    particles_beams[:, :, 1, 0] += transl[:, :, 0]

    # transl. po y
    particles_beams[:, :, 0, 1] += transl[:, :, 1]
    particles_beams[:, :, 1, 1] += transl[:, :, 1]

    particles_beams = particles_beams.reshape(-1, 2, 2)
    return particles_beams


def measurement_model(particles_beams, robot):

    # po uzoru na robot.py, 42. & 43. redak
    p_measured_res = particles_beams[:, :, 1, :] - particles_beams[:, :, 0, :]
    z_t_star = np.sqrt(p_measured_res[:, :, 0] ** 2 + p_measured_res[:, :, 1] ** 2)

    # CASE 1
    sigma_hit = robot.sensor.sigma_hit
    N = 1 / np.sqrt(2 * np.pi * sigma_hit ** 2) * np.exp(-0.5 * (robot.sensor.z_t - z_t_star) ** 2 / sigma_hit ** 2)

    # eta
    dz = 0.1
    z_max = robot.sensor.z_max
    z_arr = np.linspace(0, z_max, int(z_max / dz) + 1)
    z_tensor = np.tile(
        np.expand_dims(z_arr, axis=(1, 2)),
        (1, z_t_star.shape[0], z_t_star.shape[1])
    )
    eta_int = 1 / np.sqrt(2 * np.pi * sigma_hit ** 2) * np.exp(-0.5 * (z_tensor - z_t_star) ** 2 / sigma_hit ** 2) * dz # na vježbama sam zaboravio pomnožiti s dz
    eta = np.sum(eta_int, axis=0) ** -1

    M = particles_beams.shape[0]
    z_t_repeat = np.expand_dims(robot.sensor.z_t, axis=0).repeat(M, axis=0)
    p_hit = eta * N
    p_hit[z_t_repeat < 0] = 0
    p_hit[z_t_repeat > z_max] = 0

    # CASE 2
    lambda_short = robot.sensor.lambda_short
    eta = 1 / (1 - np.exp(-lambda_short * z_t_star))
    p_short = eta * lambda_short * np.exp(-lambda_short * robot.sensor.z_t)
    p_short[z_t_repeat < 0] = 0
    p_short[z_t_repeat > z_t_star] = 0

    # CASE 3
    z_eps = robot.sensor.z_eps
    p_max = 1 / 2 / z_eps * np.ones_like(z_t_star)
    p_max[z_t_repeat < (z_max - z_eps)] = 0
    p_max[z_t_repeat > (z_max + z_eps)] = 0 # na vježbama sam slučajno stavio z_max - z_eps, a treba ići z_max + z_eps

    # CASE 4
    p_rand = 1 / z_max * np.ones_like(z_t_star)
    p_rand[z_t_repeat < 0] = 0
    p_rand[z_t_repeat >= z_max] = 0

    z_hit = robot.sensor.z_hit
    z_short = robot.sensor.z_short
    z_maximum = robot.sensor.z_maximum
    z_rand = robot.sensor.z_rand

    p = z_hit * p_hit + z_short * p_short + z_maximum * p_max + z_rand * p_rand
    
    weights = q = np.prod(p, axis=1)
    
    return weights

if __name__ == "__main__":
    M = 1000
    particles = np.random.rand(M, 3)
    beams = np.random.rand(M, 180, 2, 2)
    particles_beams = transform_particles_beams(particles, beams)
    # pravim se da sam napravio ray casting
    particles_beams = particles_beams.reshape((M, 180, 2, 2))
    robot = Robot()
    measurement_model(particles_beams, robot)





