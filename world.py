import numpy as np

def generate_triangle(generate_from_x = -3, generate_to_x = 3, generate_from_y = -3, generate_to_y = 3):
    
    while True:

        point1 = np.array([np.random.uniform(generate_from_x, generate_to_x), 
                           np.random.uniform(generate_from_y, generate_to_y)], dtype=float)
        
        angle = np.random.uniform(0, 2 * np.pi)
        angle2 = np.random.uniform(np.pi / 4 , 3 * np.pi / 4)

        length = np.random.uniform(0.75, 1)
        length2 = np.random.uniform(0.75, 1)

        point2 = point1 + np.array([length * np.cos(angle), length * np.sin(angle)], dtype=float)
        point3 = point1 + np.array([length2 * np.cos(angle2), length2 * np.sin(angle2)], dtype=float)

        generate_range = lambda p: (generate_from_x <= p[0] <= generate_to_x) and \
                                   (generate_from_y <= p[1] <= generate_to_y)
        
        area_triangle = (1 / 2) * abs(point1[0] * (point2[1] - point3[1]) +
                                      point2[0] * (point3[1] - point1[1]) +
                                      point3[0] * (point1[1] - point2[1]))
        
        area = abs(generate_to_x - generate_from_x) * abs(generate_to_y - generate_from_y)
        
        if all(generate_range(p) for p in [point1, point2, point3]) and (0.05 * area <= area_triangle <= 0.10 * area):
        
            triangle = np.array([
                [point1, point2],
                [point2, point3],
                [point3, point1],
            ], dtype=float)
            return triangle

def generate_square(generate_from_x = -3, generate_to_x = 3, generate_from_y = -3, generate_to_y = 3):
    while True:

        point1 = np.array([
            np.random.uniform(generate_from_x, generate_to_x), 
            np.random.uniform(generate_from_y, generate_to_y)], dtype=float)

        angle = np.random.uniform(0, 2 * np.pi)

        area = abs(generate_to_x - generate_from_x) * abs(generate_to_y - generate_from_y)

        min_len = (0.05 * area) ** (1 / 2) 
        max_len = (0.10 * area) ** (1 / 2) 

        length = np.random.uniform(min_len, max_len)

        point2 = point1 + np.array([length * np.cos(angle), length * np.sin(angle)], dtype=float)
        point3 = point2 + np.array([-length * np.sin(angle), length * np.cos(angle)], dtype=float)
        point4 = point1 + np.array([-length * np.sin(angle), length * np.cos(angle)], dtype=float)

        generate_range = lambda p: (generate_from_x <= p[0] <= generate_to_x) and \
                                   (generate_from_y <= p[1] <= generate_to_y)

        if all(generate_range(p) for p in [point1, point2, point3, point4]):

            rectangle = np.array([
                [point1, point2],
                [point2, point3],
                [point3, point4],
                [point4, point1]
            ], dtype=float)
            return rectangle

def generate_hexagon(generate_from_x = -3, generate_to_x = 3, generate_from_y = -3, generate_to_y = 3):
    while True:
        point1 = np.array([np.random.uniform(generate_from_x, generate_to_x), 
                           np.random.uniform(generate_from_y, generate_to_y)], dtype=float)

        angle = np.random.uniform(0, 2 * np.pi)

        area = abs(generate_to_x - generate_from_x) * abs(generate_to_y - generate_from_y)

        min_len = ((0.10 * area * 2) / (3 * (3  ** (1 / 2)))) ** (1 / 2)
        max_len = ((0.15 * area * 2) / (3 * (3  ** (1 / 2)))) ** (1 / 2)

        length = np.random.uniform(min_len, max_len)

        point2 = point1 + np.array([length * np.cos(angle), length * np.sin(angle)], dtype=float)
        point3 = point2 + np.array([length * np.cos(angle + np.pi / 3), length * np.sin(angle + np.pi / 3)], dtype=float)
        point4 = point3 + np.array([length * np.cos(angle + 2 * np.pi / 3), length * np.sin(angle + 2 * np.pi / 3)], dtype=float)
        point5 = point4 + np.array([length * np.cos(angle + np.pi), length * np.sin(angle + np.pi)], dtype=float)
        point6 = point5 + np.array([length * np.cos(angle - 2 * np.pi / 3), length * np.sin(angle - 2 * np.pi / 3)], dtype=float)
        
        generate_range = lambda p: (generate_from_x <= p[0] <= generate_to_x) and \
                                   (generate_from_y <= p[1] <= generate_to_y)

        if all(generate_range(p) for p in [point1, point2, point3, point4, point5, point6]):

            hexagon = np.array([
                [point1, point2],
                [point2, point3],
                [point3, point4],
                [point4, point5],
                [point5, point6],
                [point6, point1]
            ], dtype=float)
            return hexagon

def generate_world():
    world = np.array([
        [[-3, -3], [-3, 3]],
        [[-3, 3], [3, 3]],
        [[3, 3], [3, -3]],
        [[3, -3], [-3, -3]]
    ], dtype=float)

    triangle = generate_triangle(-3, -1, 1, 3)
    square = generate_square(-1, 1, 1, 3)
    hexagon = generate_hexagon(1, 3, 1, 3)

    triangle2 = generate_triangle(-3, -1, -1, 1)
    square2 = generate_square(-1, 1, -1, 1)
    hexagon2 = generate_hexagon(1, 3, -1, 1)

    triangle3 = generate_triangle(-3, -1, -3, -1)
    square3 = generate_square(-1, 1, -3, -1)
    hexagon3 = generate_hexagon(1, 3, -3, -1)

    world = np.concatenate((world, 
                            triangle,  square,  hexagon, 
                            triangle2, square2, hexagon2,
                            triangle3, square3, hexagon3), axis=0)
    
    return world