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

def inside_polygon3d(points, polygon):
    num_points = points.shape[0]
    num_vertices = polygon.shape[0]

    # Calculate lengths of sides and rays
    res_sides = polygon[:, 1, :] - polygon[:, 0, :]
    l_sides = np.sqrt(res_sides[:, 0] ** 2 + res_sides[:, 1] ** 2)

    res_rays = polygon - np.mean(polygon, axis=0)
    l_rays = np.sqrt(res_rays[:, :, 0] ** 2 + res_rays[:, :, 1] ** 2)

    # Calculate area of the polygon
    a = l_sides
    b = l_rays[:, 0]
    c = l_rays[:, 1]
    s = (a + b + c) / 2
    a_r = np.sum(np.sqrt(s * (s - a) * (s - b) * (s - c)))

    # Generate triplets for each point and polygon vertex
    triplets = np.zeros((num_vertices * num_points, 3, 2))
    triplets[:, 0:2, :] = np.tile(polygon, (num_points, 1, 1))
    triplets[:, 2, :] = np.repeat(points[:, :, :2], num_vertices, axis=0)

    # Calculate areas of triangles formed by points and polygon vertices
    a = np.sqrt((triplets[:, 0, 0] - triplets[:, 1, 0]) ** 2 + (triplets[:, 0, 1] - triplets[:, 1, 1]) ** 2)
    b = np.sqrt((triplets[:, 1, 0] - triplets[:, 2, 0]) ** 2 + (triplets[:, 1, 1] - triplets[:, 2, 1]) ** 2)
    c = np.sqrt((triplets[:, 2, 0] - triplets[:, 0, 0]) ** 2 + (triplets[:, 2, 1] - triplets[:, 0, 1]) ** 2)

    s = (a + b + c) / 2
    a_p = np.sqrt(s * (s - a) * (s - b) * (s - c))
    a_p = np.sum(a_p.reshape((num_points, num_vertices)), axis=1)

    # Check if the points are inside the polygon
    return np.abs(a_p - a_r) < 1e-8


# def inside_polygon_robot(robot, polygon, robot_radius):
#     angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)

#     for angle in angles:
#         x = robot[0][0] + robot_radius * np.cos(angle)
#         y = robot[0][1] + robot_radius * np.sin(angle)

#         if inside_polygon(np.array([[x, y]], dtype=float), polygon)[0]:
#             return True  
#     return False

def inside_polygon_robot(points, lines_env, buffer):
    # Extract the x and y coordinates of the points
    x = points[:, 0]
    y = points[:, 1]
    
    # Extract the x and y coordinates of the polygon vertices
    poly_x = lines_env[:, :, 0]
    poly_y = lines_env[:, :, 1]
    
    # Compute the differences between consecutive vertices
    dx = poly_x[:, 1] - poly_x[:, 0]
    dy = poly_y[:, 1] - poly_y[:, 0]
    
    # Compute the vector products and dot products
    cross = dx * (y - poly_y[:, 0]) - dy * (x - poly_x[:, 0])
    dot = dx * (x - poly_x[:, 0]) + dy * (y - poly_y[:, 0])
    
    # Check if the points are inside the polygon
    inside = np.logical_and(dot > 0, dot < np.sum(dx**2 + dy**2))
    inside = np.logical_and(inside, np.abs(cross) < buffer)
    
    return np.any(inside, axis=0)


def near_polygon(point, polygon):
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)

    for angle in angles:
        x = point[0] + 0.2 * np.cos(angle)
        y = point[0] + 0.2 * np.sin(angle)

        if inside_polygon(np.array([[x, y]], dtype=float), polygon)[0]:
            return True  
    return False

def near_wall(point, polygon):
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)

    for angle in angles:
        x = point[0] + 0.21 * np.cos(angle)
        y = point[1] + 0.21 * np.sin(angle)

        if  not inside_polygon(np.array([[x, y]], dtype=float), polygon)[0]:
            return True  
    return False

def expand_points(points):
        expanded_points_list = []
        for point in points:
            
            angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
            
            expanded_points = np.array([[point[0] + 0.2 * np.cos(angle), point[1] + 0.2 * np.sin(angle)] for angle in angles])
            
            expanded_points_list.append(np.vstack((point, expanded_points)))

        return np.array(expanded_points_list, dtype=float)