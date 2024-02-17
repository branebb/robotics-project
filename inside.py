import numpy as np

def inside_polygon(points, polygon):
    res_sides = polygon[:, 1, :] - polygon[:, 0, :]
    l_sides = np.sqrt(res_sides[:, 0] ** 2 + res_sides[:, 1] ** 2)

    res_rays = polygon - np.mean(polygon, axis=0)
    l_rays = np.sqrt(res_rays[:, :, 0] ** 2 + res_rays[:, :, 1] ** 2)
    a = l_sides
    b = l_rays[:, 0]
    c = l_rays[:, 1]
    s = (a + b + c) / 2
    a_r = np.sum(np.sqrt(s * (s - a) * (s - b) * (s - c)))

    triplets = np.zeros((polygon.shape[0] * points.shape[0], 3, 2))
    triplets[:, 0:2, :] = np.tile(polygon, (points.shape[0], 1, 1))
    triplets[:, 2, :] = np.repeat(points[:, :2], polygon.shape[0], axis=0)

    a = np.sqrt((triplets[:, 0, 0] - triplets[:, 1, 0]) ** 2 + (triplets[:, 0, 1] - triplets[:, 1, 1]) ** 2)
    b = np.sqrt((triplets[:, 1, 0] - triplets[:, 2, 0]) ** 2 + (triplets[:, 1, 1] - triplets[:, 2, 1]) ** 2)
    c = np.sqrt((triplets[:, 2, 0] - triplets[:, 0, 0]) ** 2 + (triplets[:, 2, 1] - triplets[:, 0, 1]) ** 2)

    s = (a + b + c) / 2
    a_p = np.sqrt(s * (s - a) * (s - b) * (s - c))
    a_p = np.sum(a_p.reshape((points.shape[0], polygon.shape[0])), axis=1)

    return np.abs(a_p - a_r) < 1e-8