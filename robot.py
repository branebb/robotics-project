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

class Sensor:
    __slots__ = ["alphas_deg", "num_beams", "z_max", "S_beams", "distances", "sigma_hit", "lambda_short", "z_hit", "z_short", "z_rand", "z_maximum", "z_eps", "z_t"]
    
    def __init__(self):
        self.z_max = 3
        self.alphas_deg = np.arange(-90, 90, 5, dtype=float)
        self.num_beams = self.alphas_deg.shape[0]
        self.S_beams = np.zeros((self.num_beams, 2, 2), dtype=float)
        self.S_beams[:, 1, 0] = np.cos(np.deg2rad(self.alphas_deg)) * self.z_max
        self.S_beams[:, 1, 1] = np.sin(np.deg2rad(self.alphas_deg)) * self.z_max
        self.distances = np.ones(self.num_beams, dtype=float) * self.z_max
        
        # ovako smo inicijalizirali hiperparametre na vježbama
        self.sigma_hit = 0.05
        self.lambda_short = 2
        self.z_hit = 0.8
        self.z_short = 0.1
        self.z_rand = 0.05
        self.z_maximum = 0.05
        self.z_eps = 0.001
        
        # ovo su hiperparametri od prošle godine
        self.sigma_hit = 0.11
        self.lambda_short = 0.5

        self.z_hit = 0.8
        self.z_short = 0.1
        self.z_rand = 0.08
        self.z_maximum = 0.02
        self.z_eps = 0.001        
        
        self.z_t = self.distances.copy()

    def update_distances(self, measured):
        measured_res = measured[:, 1, :] - measured[:, 0, :]
        self.distances = np.sqrt(measured_res[:, 0] ** 2 + measured_res[:, 1] ** 2)
        
        n = self.distances.shape[0]

        case_1 = (
            self.distances + np.random.normal(0, self.sigma_hit, n)
        ).clip(min=0, max=self.z_max)

        case_2 = (
            np.random.exponential(1 / self.lambda_short, n)
        ).clip(max=self.distances)

        case_3 = np.random.uniform(
            self.z_max - self.z_eps, self.z_max + self.z_eps, n
        )

        case_4 = np.random.uniform(
            0, self.z_max, n
        )

        case_indices = np.random.choice(
            np.arange(4),
            (1, n),
            p = [self.z_hit, self.z_short, self.z_maximum, self.z_rand]
        ).T.repeat(4, axis=1)

        case_range = np.arange(0, 4).reshape(1, 4).repeat(
            n, axis=0
        )
    
        # case probabilities
        cp = (case_indices == case_range)


        self.z_t = case_1 * cp[:, 0] + case_2 * cp[:, 1] + case_3 * cp[:, 2] + case_4 * cp[:, 3]        
        
        


class Robot:
    __slots__ = ["I_xi", "l", "r", "cpr", "sensor"]
    
    def __init__(self, I_xi=np.zeros(3, dtype=float)):
        self.I_xi = I_xi
        self.l = 0.2
        self.r = 0.05
        self.cpr = 100
        
        self.sensor = Sensor()
        
    @property
    def I_sensor_full(self):
        beams = self.sensor.S_beams.copy()
        beams[:, 1, :] = np.dot(beams[:, 1, :], R(self.I_xi[2])[:2, :2])
        beams[:, :, 0] += self.I_xi[0]
        beams[:, :, 1] += self.I_xi[1]
        return beams

    @property
    def I_sensor(self):
        beams = self.sensor.S_beams.copy() / self.sensor.z_max
        beams[:, 1, 0] *= self.sensor.z_t
        beams[:, 1, 1] *= self.sensor.z_t
        beams[:, 1, :] = np.dot(beams[:, 1, :], R(self.I_xi[2])[:2, :2])
        beams[:, :, 0] += self.I_xi[0]
        beams[:, :, 1] += self.I_xi[1]
        return beams        

    def measure(self, lines_env):
        measured = ray_casting(lines_env, self.I_sensor_full)
        self.sensor.update_distances(measured)
        return self.I_sensor
        
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
    

    # u stvarnosti ovo imamo zdravo za gotovo
    # sad simuliramo
    def read_encoders(self, phis_dot, dt):
        delta_c_left = int(self.cpr * phis_dot[0] * dt / 2 / np.pi)
        delta_c_right = int(self.cpr * phis_dot[1] * dt / 2 / np.pi)
        return np.array([delta_c_left, delta_c_right], dtype=float)
    
    def enc_deltas_to_phis_dot(self, deltas_c, dt):
        phi_dot_l_enc = deltas_c[0] * 2 * np.pi / self.cpr / dt
        phi_dot_r_enc = deltas_c[1] * 2 * np.pi / self.cpr / dt
        return np.array([phi_dot_l_enc, phi_dot_r_enc, 0], dtype=float)