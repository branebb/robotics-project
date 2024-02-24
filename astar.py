#calculating heuristic value from some point to the goal
#big values - far from goal, small values - near goal
def heuristic(x, y, goal):
    return abs(goal[0] - x) + abs(goal[1] - y)

#given robot position, grid and goal position calculate path and return it
def find_path(robot_position, goal_position, grid):

    #transform real goal positions into grid positions 
    goal = [round(abs((goal_position[1] - 2.95) * 10)),
            round((goal_position[0] + 2.95) * 10)]
    
    #all movements possible
    #all directions + diagonal
    delta = [[-1, 0],
             [0, -1],
             [1, 0],
             [0, 1],
             [1, 1],
             [1, -1],
             [-1, 1],
             [-1, -1]]
    #cost per move
    cost = 1
    
    #closed/visited array for keeping track of the vertices visited
    closed = [[0 for row in range(60)] for col in range(60)]
    closed[round(abs((robot_position[1] - 2.95) * 10))][round((robot_position[0] + 2.95) * 10)] = 1

    #array for keeping track of the cost/steps to all points -1 if not possible or not discovered
    expand = [[-1 for row in range(60)] for col in range(60)]
    #parent array for recreating the ideal path
    parent = [[[-1, -1] for row in range(60)] for col in range(60)]

    #initial settings position of robot transformed, initial g value of 0
    #inital heuristic from start to finish, f value = g value + heuristic,
    #in the long run favoring the smaller values, heuristic part helps
    x = round(abs((robot_position[1] - 2.95) * 10))
    y = round((robot_position[0] + 2.95) * 10)
    g = 0
    h = heuristic(x, y, goal_position)
    f = g + h

    #constructing all informations into one array for sorting with respect of f value
    open = [[f, g, h, x, y]]

    #inital flags and count of steps
    found = False
    resign = False
    count = 0

    while found is False and resign is False:
        if len(open) == 0:
            resign = True
            print("Fail")

        else:
            open.sort()
            open.reverse()
            next = open.pop()
            g = next[1]
            x = next[3]
            y = next[4]
            expand[x][y] = count
            count += 1

            if x == goal[0] and y == goal[1]:
                found = True
            
            else:
                #adding all possible positions in all directions possible from one point
                #after for loop we repeat it sort again and then pop the next vertex
                #using A* some verticecs wont be discovered, because of the favoring ones with
                #smaller f => smaller heuristic value helps
                for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < 60 and y2 >= 0 and y2 < 60:
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            h2 = heuristic(x2, y2, goal)
                            f2 = g2 + h2
                            open.append([f2, g2, h2, x2, y2])
                            closed[x2][y2] = 1
                            parent[x2][y2] = [x, y]

    #recreating path from goal to start using parent array 
    path = []
    x, y = goal
    
    while y != round((robot_position[0] + 2.95) * 10) or x != round(abs((robot_position[1] - 2.95) * 10)):
        path.append([x, y])
        x, y = parent[x][y]

    path.append([round(abs((robot_position[1] - 2.95) * 10)), round((robot_position[0] + 2.95) * 10)])
    path.reverse()
    
    return path