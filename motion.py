import numpy as np
from shared import normalize_angle

def sample_normal_distribution(b_squared, M):
    b = np.sqrt(b_squared)

    return 1 / 2 * np.sum(
        np.random.uniform(-b, b, (M, 12)), axis=1
    )

alpha_1 = alpha_2 = alpha_3 = alpha_4 = 0.6

def sample_motion_model_odometry(u_current, x_prev):
    M = x_prev.shape[0]

    delta_rot_1 = np.arctan2(
        u_current[1, 1] - u_current[0, 1],
        u_current[1, 0] - u_current[0, 0],
    ) - u_current[0, 2]

    delta_trans = np.sqrt(
        (u_current[0, 0] - u_current[1, 0]) ** 2 + \
        (u_current[0, 1] - u_current[1, 1]) ** 2
    )

    delta_rot_2 = u_current[1, 2] - u_current[0, 2] - delta_rot_1

    delta_rot_1 = normalize_angle(delta_rot_1)
    delta_rot_2 = normalize_angle(delta_rot_2)

    delta_hat_rot_1 = delta_rot_1 - sample_normal_distribution(alpha_1 * delta_rot_1 ** 2 + alpha_2 * delta_trans ** 2, M)
    delta_hat_trans = delta_trans - sample_normal_distribution(alpha_3 * delta_trans ** 2 + alpha_4 * delta_rot_1 ** 2 + alpha_4 * delta_rot_2 ** 2, M)
    delta_hat_rot_2 = delta_rot_2 - sample_normal_distribution(alpha_1 * delta_rot_2 ** 2 + alpha_2 * delta_trans ** 2, M)

    # u prezentaciji x
    x = x_prev[:, 0]
    y = x_prev[:, 1]
    theta = x_prev[:, 2]

    x_dash = x + delta_hat_trans * np.cos(theta + delta_hat_rot_1)
    y_dash = y + delta_hat_trans * np.sin(theta + delta_hat_rot_1)
    theta_dash = theta + delta_hat_rot_1 + delta_hat_rot_2

    x_current = np.vstack((x_dash, y_dash, theta_dash)).T
    
    return x_current

if __name__ == "__main__":
    #print(sample_normal_distribution(5, 10).shape)
    I_ksi_prev = np.random.rand(3)
    I_ksi = np.random.rand(3)

    u_current = np.vstack([I_ksi_prev, I_ksi])
    M = 100
    particles = np.random.rand(M, 3)
    print(particles.shape)

    sample_motion_model_odometry(u_current, particles)
    