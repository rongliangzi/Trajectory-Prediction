import math


def counterclockwise_rotate(x, y, intersection, theta):
    # rotate the x,y point counterclockwise at angle theta around intersection point
    x_rot = (x - intersection[0]) * math.cos(theta) - (y - intersection[1]) * math.sin(theta) + intersection[0]
    y_rot = (x - intersection[0]) * math.sin(theta) + (y - intersection[1]) * math.cos(theta) + intersection[1]
    return x_rot, y_rot


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

    if angle > math.pi/4 and closest_point_id < len(path_points)-1:
        closest_point_id += 1

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

    # judge the sign of d
    len_d = distance(x_x, x_y, proj_x, proj_y)
    len_pre_next = (n_x ** 2 + n_y ** 2) ** 0.5
    d_in_heading = (proj_x + len_d * n_x / len_pre_next, proj_y + len_d * n_y / len_pre_next)
    # see if d value is positive or negative by comparing it to a center point
    x_rot1, y_rot1 = counterclockwise_rotate(d_in_heading[0], d_in_heading[1], (proj_x, proj_y), math.pi/2)
    x_rot3, y_rot3 = counterclockwise_rotate(d_in_heading[0], d_in_heading[1], (proj_x, proj_y), math.pi * 3 / 2)
    dis1 = distance(x_rot1, y_rot1, x_x, x_y)
    dis3 = distance(x_rot3, y_rot3, x_x, x_y)
    if dis1 < dis3:
        frenet_d = len_d
    else:
        frenet_d = -len_d
    # get s value from pre calculated list
    frenet_s = frenet_s_list[prev_wp]
    frenet_s += distance(0, 0, proj_x, proj_y)

    return frenet_s, frenet_d, (proj_x+path_points[prev_wp][0], proj_y+path_points[prev_wp][1])


# Transform from Frenet s,d coordinates to Cartesian x,y
def get_xy(s, d, path_s, path_wp, proj):
    prev_wp = -1
    while prev_wp < len(path_s) - 1 and s > path_s[prev_wp + 1]:
        prev_wp += 1
    if prev_wp == len(path_s)-1:
        prev_wp = len(path_s) - 2
    next_wp = prev_wp + 1
    heading = math.atan2((path_wp[next_wp][1] - path_wp[prev_wp][1]), (path_wp[next_wp][0]-path_wp[prev_wp][0]))
    # the x,y,s along the segment
    seg_s = s - path_s[prev_wp]
    # seg_x = path_wp[prev_wp][0] + seg_s * math.cos(heading)
    # seg_y = path_wp[prev_wp][1] + seg_s * math.sin(heading)
    seg_x, seg_y = proj
    d_in_heading = (seg_x + abs(d) * math.cos(heading), seg_y + abs(d) * math.sin(heading))
    if d > 0:
        x, y = counterclockwise_rotate(d_in_heading[0], d_in_heading[1], (seg_x, seg_y), math.pi / 2)
    else:
        x, y = counterclockwise_rotate(d_in_heading[0], d_in_heading[1], (seg_x, seg_y), math.pi * 3 / 2)

    return x, y
