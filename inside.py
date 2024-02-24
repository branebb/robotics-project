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

#checking if robot or some point in some radius is in the polygon
#True if only one point is in, otherwise False
def inside_polygon_robot(robot, polygon, robot_radius):

    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)

    x = robot[0][0] + robot_radius * np.cos(angles)
    y = robot[0][1] + robot_radius * np.sin(angles)

    points = np.column_stack((x, y))

    return np.any(inside_polygon(points, polygon))

#expanding point with new axis making (3600, 136, 2) shape instead of (3600, 2)
#generating 135 additional points
def expand_points(points):
        angles = np.linspace(0, 2 * np.pi, 135, endpoint=False)

        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        cos_expanded = cos_angles.reshape(1, -1)
        sin_expanded = sin_angles.reshape(1, -1)

        expanded_points_x = points[:, 0][:, np.newaxis] + cos_expanded * 0.21
        expanded_points_y = points[:, 1][:, np.newaxis] + sin_expanded * 0.21

        expanded_points = np.stack((expanded_points_x, expanded_points_y), axis=-1)

        return expanded_points

#checking if any of the 136 points is in the one obstacle returing True is one is and repeating that
#for all obstacles resulting True/False  array with shape (3600,)
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