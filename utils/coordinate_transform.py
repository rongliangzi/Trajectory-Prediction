import math
import numpy as np
from utils.intersection_utils import cal_dis


def counterclockwise_rotate(x, y, intersection, theta):
    # rotate the x,y point counterclockwise at angle theta around intersection point
    x_rot = (x - intersection[0]) * math.cos(theta) - (y - intersection[1]) * math.sin(theta) + intersection[0]
    y_rot = (x - intersection[0]) * math.sin(theta) + (y - intersection[1]) * math.cos(theta) + intersection[1]
    return x_rot, y_rot


def distance(x1, y1, x2, y2):
    return math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))


def closest_point_index(x, y, path_points):
    dis = cal_dis(np.array([[x, y]]), np.array(path_points))
    id2 = np.argmin(dis, axis=1)
    return id2[0]


def next_point_id(x, y, path_points):
    closest_point_id = closest_point_index(x, y, path_points)
    return closest_point_id


# Transform from Cartesian x,y coordinates to Frenet s,d coordinates
def get_frenet(x, y, path_points, frenet_s_list):
    next_wp = next_point_id(x, y, path_points)
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
    drop_head = 0
    if prev_wp == 0 and proj_x * n_x < 0:
        drop_head = 1
    drop_tail = 0
    if next_wp > 0.97*len(path_points) and \
            (abs(proj_x) > 0.05 or abs(proj_y > 0.05) or abs(x_x) > 0.5 or abs(x_y) > 0.5):
        drop_tail = 1
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

    return frenet_s, frenet_d, (proj_x+path_points[prev_wp][0], proj_y+path_points[prev_wp][1]), drop_head, drop_tail


# Transform from Frenet s,d coordinates to Cartesian x,y
def get_xy(s, d, path_s, path_wp, proj=None):
    # binary search, find the prev<s<prev+1, the first prev+1>s
    prev_wp = 0
    next_wp = len(path_s) - 1
    while prev_wp <= next_wp:
        mid = (prev_wp + next_wp) // 2
        if s >= path_s[mid]:
            prev_wp = mid + 1
        else:
            next_wp = mid - 1

    prev_wp -= 1
    next_wp = prev_wp + 1
    heading = math.atan2((path_wp[next_wp][1] - path_wp[prev_wp][1]), (path_wp[next_wp][0]-path_wp[prev_wp][0]))
    # the x,y,s along the segment
    if proj is None:
        seg_s = s - path_s[prev_wp]
        seg_x = path_wp[prev_wp][0] + seg_s * math.cos(heading)
        seg_y = path_wp[prev_wp][1] + seg_s * math.sin(heading)
    else:
        seg_x, seg_y = proj
    d_in_heading = (seg_x + abs(d) * math.cos(heading), seg_y + abs(d) * math.sin(heading))
    if d > 0:
        x, y = counterclockwise_rotate(d_in_heading[0], d_in_heading[1], (seg_x, seg_y), math.pi / 2)
    else:
        x, y = counterclockwise_rotate(d_in_heading[0], d_in_heading[1], (seg_x, seg_y), math.pi * 3 / 2)

    return x, y
