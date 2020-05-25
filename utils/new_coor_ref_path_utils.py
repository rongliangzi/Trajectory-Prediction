from utils import dataset_reader
from utils import dict_utils
from utils.starting_area_utils import *
import numpy as np
import math
import glob
import os
import random
import pickle


def judge_in_box(x, y, p):
    # judge if p in the polygon formed by x y
    assert len(x) == len(y)
    xl = len(x)
    for i in range(xl-2):
        same_side = judge_point_side((x[i], y[i]), (x[i+1], y[i+1]), (x[i+2], y[i+2]), p)
        if same_side == 0:
            return 0
    same_side = judge_point_side((x[xl-2], y[xl-2]), (x[xl-1], y[xl-1]), (x[0], y[0]), p)
    if same_side == 0:
        return 0
    same_side = judge_point_side((x[xl-1], y[xl-1]), (x[0], y[0]), (x[1], y[1]), p)
    if same_side == 0:
        return 0
    return 1


def judge_point_side(p1, p2, p3, p4):
    # judge if p3 p4 are in the same side of p1 p2
    if p1[0] == p2[0]:
        if p3[0] > p1[0] > p4[0] or p4[0] > p1[0] > p3[0]:
            return 0
        else:
            return 1
    else:
        k = (p2[1]-p1[1])/(p2[0]-p1[0])
        b = (p2[0]*p1[1]-p1[0]*p2[1])/(p2[0]-p1[0])
        y3 = p3[0] * k + b
        y4 = p4[0] * k + b
        if (y3 < p3[1] and y4 < p4[1]) or (y3 > p3[1] and y4 > p4[1]):
            return 1
        else:
            return 0


def judge_start_new(ms, track):
    # judge if in some starting area frame by frame
    for ts in range(track.time_stamp_ms_first, track.time_stamp_ms_last+100, 100):
        motion_state = track.motion_states[ts]
        cur_p = (motion_state.x, motion_state.y)
        for k, v in new_coor_starting_area_dict.items():
            in_box = judge_in_box(v['x'], v['y'], cur_p)
            if in_box == 1:
                return k
    return 0


def judge_end_new(ms, track):
    if ms.x < 145 and abs(ms.y-285) < 15 and ms.vx < -0.02:
        return 1
    elif ms.y < 150 and ms.vy < 0:
        if ms.x < 180:
            return 15
        elif ms.x < 210:
            return 4
        elif 210 < ms.x:
            return 5
    elif ms.x > 370 and ms.vx > 0:
        return 8
    elif ms.y > 350 and ms.x > 255 and ms.vy > 0:
        return 10
    elif ms.y > 350 and ms.x <= 255 and ms.vy > 0:
        return 11
    else:
        return 0


def fit_ref_path(xy, k, path_w, beta_dict, poly_dict):
    x = np.array([p[0] for p in xy])
    y = np.array([p[1] for p in xy])
    beta = beta_dict[k]
    poly = poly_dict[k]
    # rotate the points clockwise
    x_rot = x * math.cos(beta) + y * math.sin(beta)
    y_rot = y * math.cos(beta) - x * math.sin(beta)
    # fit a polynomial function using rotated points
    z1 = np.polyfit(x_rot, y_rot, poly)
    p1 = np.poly1d(z1)
    x_new = np.arange(min(x_rot), max(x_rot), 0.2)
    pp1 = p1(x_new)
    # rotate the points back
    y_back = pp1 * math.cos(beta) + x_new * math.sin(beta)
    x_back = x_new * math.cos(beta) - pp1 * math.sin(beta)
    # mask far away points
    if k == '7-8':
        # y_mask = y_back > 965
        # x_mask = x_back < 1080
        y_mask = y_back > 75
        x_mask = x_back < 595
        mask = (x_mask.astype(np.int) + y_mask.astype(np.int)) > 1
    elif k == '12-8':
        # mask = (y_back < 1035) & (x_back < 1090)
        mask = (y_back < 425) & (x_back < 595)
    elif k == '9-5':
        # mask = (y_back > 968) & (y_back < 1035) & (x_back > 970) & (x_back < 1090)
        mask = (y_back > 90) & (y_back < 425) & (x_back > -1) & (x_back < 595)
    else:
        # mask = (y_back > 965) & (y_back < 1035) & (x_back > 972.4) & (x_back < 1090)
        mask = (y_back > 75) & (y_back < 425) & (x_back > 7) & (x_back < 595)
    y_back = y_back[mask]
    x_back = x_back[mask]
    path_w = np.array(path_w)
    avg_path_w = np.average(path_w)

    # reverse the direction
    if k in ['6-1', '7-10', '9-1', '9-4', '9-5', '9-10', '6-4', '7-11', '6-10', '6-11', '9-11', '13-4', '14-1', '12-5']:
        x_back = x_back[::-1]
        y_back = y_back[::-1]

    poly_func = ''
    return [x_back, y_back, x, y, avg_path_w], poly_func


def get_ref_paths(base_path, dir_name):
    '''
    :param base_path:
    :param dir_name:
    :param save_pickle:
    :return: all ref paths in a dict. each item includes [x, y, raw x, raw y, path w], poly func in str
    '''
    trajectory = dict()
    csv_dict = dict()
    # collect data to construct a dict from all csv
    paths = glob.glob(os.path.join(base_path + dir_name, '*.csv'))
    paths.sort()
    for csv_name in paths:
        track_dictionary = dataset_reader.read_tracks(csv_name)
        tracks = dict_utils.get_value_list(track_dictionary)
        agent_path = list()
        for agent in tracks:
            # transform the coordinate
            for ts in range(agent.time_stamp_ms_first, agent.time_stamp_ms_last+100, 100):
                agent.motion_states[ts].x = (agent.motion_states[ts].x - x_s) * rate
                agent.motion_states[ts].y = (agent.motion_states[ts].y - y_s) * rate
            first_ms = agent.motion_states[agent.time_stamp_ms_first]
            last_ms = agent.motion_states[agent.time_stamp_ms_last]
            start_area = judge_start_new(first_ms, agent)
            end_area = judge_end_new(last_ms, agent)
            if start_area == 0 or end_area == 0:
                pass
            else:
                k = str(start_area) + '-' + str(end_area)
                agent_path.append([agent, k])
                if k not in trajectory:
                    trajectory[k] = [agent]
                elif k not in ['12-10', '13-10', '14-10']:
                    trajectory[k].append(agent)
        csv_dict[csv_name[-7:-4]] = agent_path
    ref_paths = dict()
    beta_dict = {'7-8': math.pi / 4, '12-8': -math.pi / 6, '2-10': math.pi / 4, '2-11': math.pi / 4,
                 '6-1': -math.pi / 6, '3-4': -math.pi / 6, '9-4': math.pi / 4, '9-5': math.pi / 4,
                 '9-10': -math.pi / 6, '13-5': -math.pi / 6, '14-4': -math.pi / 6, '6-11': -math.pi / 6,
                 '7-10': 0, '2-8': 0, '3-8': 0, '9-1': 0, '12-5': math.pi / 5, '13-8': -math.pi / 4,
                 '14-1': math.pi / 4, '14-15': -math.pi / 6, '13-4': math.pi / 5, '14-5': math.pi / 4,
                 '12-10': math.pi / 12, '7-11': -math.pi / 6, '9-11': -math.pi / 6, '13-10': math.pi / 12,
                 '2-4': -math.pi / 6, '3-5': -math.pi / 6, '6-10': -math.pi / 6, '6-4': 0}
    poly_dict = {'7-8': 5, '12-8': 6, '2-10': 6, '2-11': 6, '6-1': 6, '3-4': 6, '9-4': 7, '9-5': 7,
                 '9-10': 4, '13-5': 1, '14-4': 1, '6-11': 3, '7-10': 1, '2-8': 4, '3-8': 4, '9-1': 4,
                 '12-5': 3, '13-8': 4, '14-1': 5, '14-15': 1, '13-4': 3, '14-5': 6, '12-10': 4, '7-11': 4,
                 '9-11': 4, '13-10': 6, '2-4': 4, '3-5': 4, '6-10': 4, '6-4': 6}
    # for trajectories in one ref path
    rare_paths = []
    for ref_i, (k, v) in enumerate(sorted(trajectory.items())):
        xy = []
        path_w = []
        # save all (x,y) points in xy
        for track in v:
            path_w.append(track.width)
            for ts in range(track.time_stamp_ms_first, track.time_stamp_ms_last+100, 100):
                motion_state = track.motion_states[ts]
                # if (k == '12-8' and motion_state.x < 1015) or (k == '14-1' and motion_state.x > 1011):
                if (k == '12-8' and motion_state.x < 220) or (k == '14-1' and motion_state.x > 200):
                    pass
                elif k in ['7-8', '9-5'] and motion_state.y < 100:
                    # add some data points
                    for i in range(20):
                        xy.append([motion_state.x + random.random() * 0.5,
                                   motion_state.y + (random.random() - 0.5) * 0.4])
                        xy.append([motion_state.x + random.random() * 0.5,
                                   motion_state.y + random.random() - 6])
                elif k == '12-8' and motion_state.x > 520:
                    # add some data points
                    for i in range(30):
                        r = random.random() * 3
                        xy.append([motion_state.x + r,
                                   motion_state.y + r * 0.1 + random.random() * 0.8])
                else:
                    xy.append([motion_state.x, motion_state.y])
        # for rare paths, use raw points and interpolation points
        if len(v) < 2:
            print('rare path:', k)
            rare_paths.append(k)
            x_points = []
            y_points = []
            for i, point in enumerate(xy[:-1]):
                if point[0] == xy[i+1][0] and point[1] == xy[i+1][1]:
                    continue
                x_points.append(point[0])
                x_points.append((point[0]+xy[i+1][0])/2)
                y_points.append(point[1])
                y_points.append((point[1]+xy[i+1][1])/2)
            ref_paths[k] = [np.array(x_points), np.array(y_points), np.array([point[0] for point in xy]),
                            np.array([point[1] for point in xy]), v[0].width]
        else:
            ref_path, poly_func = fit_ref_path(xy, k, path_w, beta_dict, poly_dict)
            ref_paths[k] = ref_path

    return ref_paths, csv_dict, rare_paths
