import math


def distance(x1, y1, x2, y2):
    return math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))


def closest_point_index(x, y, path_points):
    closest_dis = 1e8  # large number
    closest_point_id = 0

    for i in range(len(path_points)):
        map_x = path_points[i][0]
        map_y = path_points[i][1]
        dist = distance(x, y, map_x, map_y)
        if dist < closest_dis:
            closest_dis = dist
            closest_point_id = i

    return closest_point_id


def next_point_id(x, y, psi, path_points):
    closest_point_id = closest_point_index(x, y, path_points)

    map_x = path_points[closest_point_id][0]
    map_y = path_points[closest_point_id][1]

    heading = math.atan2((map_y-y), (map_x-x))

    angle = math.fabs(psi-heading)
    angle = min(2*math.pi - angle, angle)

    if angle > math.pi/4:
        closest_point_id += 1
        if closest_point_id == len(path_points):
            closest_point_id = 0

    return closest_point_id


# Transform from Cartesian x,y coordinates to Frenet s,d coordinates
def get_frenet(x, y, psi, path_points, frenet_s_list):
    next_wp = next_point_id(x, y, psi, path_points)
    prev_wp = next_wp-1
    if next_wp == 0:
        # prev_wp = len(path_points)-1
        prev_wp = 0
        next_wp = 1
    while path_points[next_wp][0] == path_points[prev_wp][0] and path_points[next_wp][1] == path_points[prev_wp][1]:
        prev_wp -= 1
    n_x = path_points[next_wp][0] - path_points[prev_wp][0]
    n_y = path_points[next_wp][1] - path_points[prev_wp][1]
    x_x = x - path_points[prev_wp][0]
    x_y = y - path_points[prev_wp][1]

    # find the projection of x onto n

    proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y)
    proj_x = proj_norm*n_x
    proj_y = proj_norm*n_y
    # todo, judge the sign of d
    frenet_d = distance(x_x, x_y, proj_x, proj_y)

    # see if d value is positive or negative by comparing it to a center point

    center_x = 1000 - path_points[prev_wp][0]
    center_y = 2000 - path_points[prev_wp][1]
    center_to_pos = distance(center_x, center_y, x_x, x_y)
    center_to_ref = distance(center_x, center_y, proj_x, proj_y)

    if center_to_pos <= center_to_ref:
        frenet_d *= -1

    # # calculate s value
    # frenet_s = 0
    # for i in range(prev_wp):
    #     frenet_s += distance(path_points[i][0], path_points[i][1], path_points[i+1][0], path_points[i+1][1])
    #

    # get s value from pre calculated list
    frenet_s = frenet_s_list[prev_wp]
    frenet_s += distance(0, 0, proj_x, proj_y)

    return frenet_s, frenet_d
