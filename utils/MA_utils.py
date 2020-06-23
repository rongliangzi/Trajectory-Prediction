from utils import dataset_reader
from utils import dict_utils
from utils.roundabout_utils import judge_start, judge_end
import numpy as np
import math
import glob
import os
import random
import csv


def fit_ref_path(xy, k, beta_dict, poly_dict):
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

    # reverse the direction
    if k in ['6-1', '7-10', '9-1', '9-4', '9-5', '9-10', '6-4', '7-11', '6-10', '6-11', '9-11', '13-4', '14-1', '12-5']:
        x_back = x_back[::-1]
        y_back = y_back[::-1]

    return x_back, y_back


def get_ref_paths(base_path, dir_name, starting_areas, end_areas, x_s, y_s):
    trajectory = dict()
    csv_dict = dict()
    # collect data to construct a dict from all csv files
    paths = glob.glob(os.path.join(base_path + dir_name, '*.csv'))
    paths.sort()
    for csv_name in paths:
        track_dictionary = dataset_reader.read_tracks(csv_name)
        tracks = dict_utils.get_value_list(track_dictionary)
        agent_path = dict()
        for agent in tracks:
            # transform the coordinate
            for ts in range(agent.time_stamp_ms_first, agent.time_stamp_ms_last + 100, 100):
                agent.motion_states[ts].x = (agent.motion_states[ts].x - x_s)
                agent.motion_states[ts].y = (agent.motion_states[ts].y - y_s)
            start_area = judge_start(agent, starting_areas)
            end_area = judge_end(agent, end_areas)
            if start_area == 0 or end_area == 0:
                print(agent.track_id, 'starting or ending area is 0, drop')
                continue
            path_name = str(start_area) + '-' + str(end_area)
            agent_path[agent.track_id] = agent
            if path_name not in trajectory:
                trajectory[path_name] = [agent]
            elif path_name not in ['12-10', '13-10', '14-10']:
                trajectory[path_name].append(agent)
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
    for ref_i, (path_name, v) in enumerate(sorted(trajectory.items())):
        xy = []
        path_w = []
        # save all (x,y) points in xy
        for track in v:
            path_w.append(track.width)
            for ts in range(track.time_stamp_ms_first, track.time_stamp_ms_last+100, 100):
                motion_state = track.motion_states[ts]
                if (path_name == '12-8' and motion_state.x < 1015-x_s) \
                        or (path_name == '14-1' and motion_state.x > 1011-x_s):
                    pass
                elif path_name in ['7-8', '9-5'] and motion_state.y < 970-y_s:
                    # add some data points
                    for i in range(20):
                        xy.append([motion_state.x + random.random() * 0.5,
                                   motion_state.y + (random.random() - 0.5) * 0.4])
                        xy.append([motion_state.x + random.random() * 0.5,
                                   motion_state.y + random.random() - 6])
                elif path_name == '12-8' and motion_state.x > 1075-x_s:
                    # add some data points
                    for i in range(30):
                        r = random.random() * 3
                        xy.append([motion_state.x + r,
                                   motion_state.y + r * 0.1 + random.random() * 0.8])
                else:
                    xy.append([motion_state.x, motion_state.y])
        # for rare paths, use raw points and interpolation points
        if len(v) < 2:
            print('rare path:', path_name)
            rare_paths.append(path_name)
            # x_points = []
            # y_points = []
            # for i, point in enumerate(xy[:-1]):
            #     if point[0] == xy[i+1][0] and point[1] == xy[i+1][1]:
            #         continue
            #     x_points.append(point[0])
            #     x_points.append((point[0]+xy[i+1][0])/2)
            #     y_points.append(point[1])
            #     y_points.append((point[1]+xy[i+1][1])/2)
            # ref_paths[k] = [np.array(x_points), np.array(y_points), np.array([point[0] for point in xy]),
            #                 np.array([point[1] for point in xy]), v[0].width]
        else:
            ref_x_pts, ref_y_pts = fit_ref_path(xy, path_name, beta_dict, poly_dict)
            ref_x_pts += x_s
            ref_y_pts += y_s
            # nd array shape: (x,2)
            ref_paths[path_name] = np.hstack((ref_x_pts, ref_y_pts))

    return ref_paths, csv_dict, rare_paths


def ln_func(coef, v):
    assert len(coef) == 4, 'coef must have length 4, but has {}'.format(len(coef))
    result = (coef[0] + coef[1] * np.log(coef[2] * v - coef[3]))
    result = np.array(result)
    return result


def get_defined_ref_paths(defined_file, csv_dir, x_start, y_start):
    csv_dict = dict()
    paths = glob.glob(os.path.join(csv_dir, '*.csv'))
    paths.sort()
    for csv_name in paths:
        track_dictionary = dataset_reader.read_tracks(csv_name)
        tracks = dict_utils.get_value_list(track_dictionary)
        agent_path = dict()
        for agent in tracks:
            # transform the coordinate
            for ts in range(agent.time_stamp_ms_first, agent.time_stamp_ms_last + 100, 100):
                agent.motion_states[ts].x = (agent.motion_states[ts].x - x_start) * rate
                agent.motion_states[ts].y = (agent.motion_states[ts].y - y_start) * rate
            agent_path[agent.track_id] = agent
        csv_dict[csv_name[-7:-4]] = agent_path
    ref_paths = dict()
    with open(defined_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(list(csv_reader)[1:]):
            ref_path_id = int(row[1])
            xy_flag = int(float(row[5]))
            ln_flag = row[6]
            coef = []
            x_min = float(row[7])
            x_max = float(row[8])
            y_min = float(row[9])
            y_max = float(row[10])
            sp = row[4][1:-1].split(', ')
            for s in sp:
                if s:
                    coef += [float(s)]
            # coef = coef[::-1]
            if ln_flag == 'False':
                poly_func = np.poly1d(coef)
                if xy_flag == 1:
                    ref_path_y = np.arange(y_min, y_max, 0.2)
                    ref_path_x = poly_func(ref_path_y)
                else:
                    ref_path_x = np.arange(x_min, x_max, 0.2)
                    ref_path_y = poly_func(ref_path_x)
            else:
                if xy_flag == 1:
                    ref_path_y = np.arange(y_min, y_max, 0.2)
                    ref_path_x = ln_func(coef, ref_path_y)
                else:
                    ref_path_x = np.arange(x_min, x_max, 0.2)
                    ref_path_y = ln_func(coef, ref_path_x)
            ref_paths[ref_path_id] = [ref_path_x, ref_path_y, None, None, None]
    return ref_paths, csv_dict
