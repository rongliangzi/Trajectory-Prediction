from utils import dataset_reader
from utils import dict_utils
from utils.roundabout_utils import judge_start, judge_end
from utils.coordinate_transform import get_frenet
from utils import map_vis_without_lanelet
from utils.intersection_utils import find_intersection_ita
import numpy as np
import math
import glob
import os
import random
import csv
import matplotlib.pyplot as plt


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
    x_new = np.arange(min(x_rot), max(x_rot), 0.05)
    pp1 = p1(x_new)
    # rotate the points back
    y_back = pp1 * math.cos(beta) + x_new * math.sin(beta)
    x_back = x_new * math.cos(beta) - pp1 * math.sin(beta)
    # mask far away points
    if k == '7-8':
        # y_mask = y_back > 965
        # x_mask = x_back < 1075
        y_mask = y_back > 15
        x_mask = x_back < 110
        mask = (x_mask.astype(np.int) + y_mask.astype(np.int)) > 1
    elif k == '12-8':
        mask = (y_back < 85) & (x_back < 110)
    elif k == '9-5':
        mask = (y_back > 18) & (y_back < 85) & (x_back > -1) & (x_back < 110)
    else:
        mask = (y_back > 15) & (y_back < 85) & (x_back > 1.4) & (x_back < 110)
    y_back = y_back[mask]
    x_back = x_back[mask]

    # reverse the direction
    if k in ['6-1', '7-10', '9-1', '9-4', '9-5', '9-10', '6-4', '7-11', '6-10', '6-11',
             '9-11', '13-4', '14-1', '12-5', '6-5']:
        x_back = x_back[::-1]
        y_back = y_back[::-1]

    return x_back, y_back


def get_ref_paths(base_path, dir_name, starting_areas, end_areas, x_s, y_s, save_img=False):
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
            # judge start and end area
            start_area = judge_start(agent, starting_areas)
            end_area = judge_end(agent, end_areas)
            if start_area == 0 or end_area == 0:
                # print(agent.track_id, 'starting or ending area is 0, discard')
                continue
            path_name = str(start_area) + '-' + str(end_area)
            agent.ref_path_id = path_name
            agent_path[agent.track_id] = agent
            if path_name not in trajectory:
                trajectory[path_name] = [agent]
            elif path_name not in ['12-10', '13-10', '14-10', '2-1']:
                trajectory[path_name].append(agent)
        csv_dict[csv_name[-7:-4]] = agent_path
    ref_paths = dict()
    beta_dict = {'7-8': math.pi / 4, '12-8': -math.pi / 6, '2-10': math.pi / 4, '2-11': math.pi / 4,
                 '6-1': -math.pi / 6, '3-4': -math.pi / 6, '9-4': math.pi / 4, '9-5': math.pi / 4,
                 '9-10': -math.pi / 6, '13-5': -math.pi / 6, '14-4': -math.pi / 6, '6-11': -math.pi / 6,
                 '7-10': 0, '2-8': 0, '3-8': 0, '9-1': 0, '12-5': math.pi / 5, '13-8': -math.pi / 4,
                 '14-1': math.pi / 4, '14-15': -math.pi / 6, '13-4': math.pi / 5, '14-5': math.pi / 4,
                 '12-10': math.pi / 12, '7-11': -math.pi / 6, '9-11': -math.pi / 6, '13-10': math.pi / 12,
                 '2-4': -math.pi / 6, '3-5': -math.pi / 6, '6-10': -math.pi / 6, '6-4': 0, '6-5': 0,
                 '3-15': -math.pi / 6}
    poly_dict = {'7-8': 5, '12-8': 6, '2-10': 6, '2-11': 6, '6-1': 6, '3-4': 6, '9-4': 7, '9-5': 7,
                 '9-10': 4, '13-5': 1, '14-4': 1, '6-11': 3, '7-10': 1, '2-8': 4, '3-8': 4, '9-1': 4,
                 '12-5': 3, '13-8': 4, '14-1': 5, '14-15': 1, '13-4': 3, '14-5': 6, '12-10': 4, '7-11': 4,
                 '9-11': 4, '13-10': 6, '2-4': 4, '3-5': 4, '6-10': 4, '6-4': 6, '6-5': 6, '3-15': 4}
    rare_paths = []
    # for trajectories in one ref path
    for ref_i, (path_name, v) in enumerate(sorted(trajectory.items())):
        xy = []
        # save all (x,y) points in xy
        for track in v:
            for ts in range(track.time_stamp_ms_first, track.time_stamp_ms_last+100, 100):
                motion_state = track.motion_states[ts]
                if (path_name == '12-8' and motion_state.x < 1015) \
                        or (path_name == '14-1' and motion_state.x > 1011):
                    pass
                elif path_name in ['7-8', '9-5'] and motion_state.y < 970:
                    # add some data points
                    for i in range(20):
                        xy.append([motion_state.x - x_s + random.random() * 0.5,
                                   motion_state.y - y_s + (random.random() - 0.5) * 0.4])
                        xy.append([motion_state.x - x_s + random.random() * 0.5,
                                   motion_state.y - y_s + random.random() - 6])
                elif path_name == '12-8' and motion_state.x > 1075:
                    # add some data points
                    for i in range(30):
                        r = random.random() * 3
                        xy.append([motion_state.x - x_s + r,
                                   motion_state.y - y_s + r * 0.1 + random.random() * 0.8])
                else:
                    xy.append([motion_state.x - x_s, motion_state.y - y_s])
        # for rare paths, use raw points and interpolation points
        if path_name in ['12-10', '12-5', '6-4', '2-1', '3-15', '3-5', '6-10',
                         '7-11', '9-11', '9-15', '14-8', '9-8']:
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
            ref_x_pts = ref_x_pts.reshape(len(ref_x_pts), 1)
            ref_y_pts = ref_y_pts.reshape(len(ref_y_pts), 1)
            # nd array shape: (x,2)
            ref_paths[path_name] = np.hstack((ref_x_pts, ref_y_pts))
            if save_img:
                fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
                map_file = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/DR_USA_Intersection_MA.osm'
                map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
                plt.scatter([xy_[0]+x_s for xy_ in xy], [xy_[1]+y_s for xy_ in xy], s=1)
                plt.plot(ref_x_pts, ref_y_pts, linewidth=6, color='b', alpha=0.5)
                plt.text(min(ref_x_pts[0], 1084), ref_y_pts[0], 'start', zorder=30, fontsize=30)
                plt.text(min(ref_x_pts[-1], 1084), ref_y_pts[-1], 'end', zorder=30, fontsize=30)
                plt.savefig('D:/Dev/UCB task/path_imgs/MA/{}.png'.format(path_name))
                plt.close()
    return ref_paths, csv_dict, rare_paths


def get_track_label(csv_data, ref_path_points, ref_frenet, rare_paths):
    all_csv_dict = dict()
    print('get_track_label:')
    for csv_id, csv_agents in csv_data.items():
        print(csv_id)
        csv_dict = dict()
        for agent_id, agent in csv_agents.items():
            path_name = agent.ref_path_id
            if path_name in rare_paths:
                continue
            agent_dict = dict()
            agent_dict['track_id'] = agent.track_id

            agent_dict['ref path'] = path_name
            xy_points = ref_path_points[path_name]
            agent_dict['motion_states'] = dict()
            start_ts = -1
            for ts in range(agent.time_stamp_ms_first, agent.time_stamp_ms_last + 100, 100):
                ms = agent.motion_states[ts]
                x = ms.x
                y = ms.y
                _, _, _, drop_flag = get_frenet(x, y, xy_points, ref_frenet[path_name])
                if drop_flag == 0:
                    start_ts = ts
                    break
            if start_ts == -1:
                print('start_ts:-1', csv_id, agent_id, path_name)
            agent_dict['time_stamp_ms_first'] = start_ts
            agent_dict['time_stamp_ms_last'] = agent.time_stamp_ms_last
            for ts in range(agent_dict['time_stamp_ms_first'], agent_dict['time_stamp_ms_last'] + 100, 100):
                ms = agent.motion_states[ts]
                agent_dict['motion_states'][ts] = dict()
                agent_dict['motion_states'][ts]['time_stamp_ms'] = ms.time_stamp_ms
                x = agent_dict['motion_states'][ts]['x'] = ms.x
                y = agent_dict['motion_states'][ts]['y'] = ms.y
                agent_dict['motion_states'][ts]['vx'] = ms.vx
                agent_dict['motion_states'][ts]['vy'] = ms.vy
                agent_dict['motion_states'][ts]['psi_rad'] = ms.psi_rad
                f_s, f_d, proj, drop_flag = get_frenet(x, y, xy_points, ref_frenet[path_name])
                agent_dict['motion_states'][ts]['frenet_s'] = f_s
                agent_dict['motion_states'][ts]['frenet_d'] = f_d
                agent_dict['motion_states'][ts]['proj'] = proj
                if ts > agent_dict['time_stamp_ms_first']:
                    vs = (f_s - agent_dict['motion_states'][ts-100]['frenet_s']) / 0.1
                    agent_dict['motion_states'][ts]['vs'] = vs
            csv_dict[agent_id] = agent_dict
        all_csv_dict[csv_id] = csv_dict
    return all_csv_dict


def find_ma_interactions(ref_paths, th=1, skip=60):
    interactions = dict()
    path_names = sorted(ref_paths.keys())
    for path_name in path_names:
        interactions[path_name] = dict()
    for i, path1 in enumerate(path_names):
        # if path1!='13-8':
        #     continue
        for j in range(i+1, len(path_names)):
            path2 = path_names[j]
            # if path2!='3-8':
            #     continue
            if path1 == '2-8':
                ita12, ita21 = find_intersection_ita(path1, path2, ref_paths[path1],
                                                     ref_paths[path2], 1.5, skip, m=1.1)
            elif path1 == '7-10':
                ita12, ita21 = find_intersection_ita(path1, path2, ref_paths[path1],
                                                     ref_paths[path2], 1.5, skip, m=1.1, k=2)
            elif path1 == '7-11' and path2 == '7-8':
                ita12, ita21 = find_intersection_ita(path1, path2, ref_paths[path1],
                                                     ref_paths[path2], 20, skip, dis_th=6, m=1.2, k=4)
            elif path1 == '12-8' and path2 == '3-8':
                ita12, ita21 = find_intersection_ita(path1, path2, ref_paths[path1],
                                                     ref_paths[path2], 1.5, skip, m=1.2)
            elif path1 == '13-8' and path2 == '3-8':
                ita12, ita21 = find_intersection_ita(path1, path2, ref_paths[path1],
                                                     ref_paths[path2], 1.5, skip, m=1.2)
            else:
                ita12, ita21 = find_intersection_ita(path1, path2, ref_paths[path1],
                                                     ref_paths[path2], th, skip)
            if ita12 is not None and len(ita12) > 0:
                # interaction of path1 and path2 exists
                interactions[path1][path2] = ita12
                interactions[path2][path1] = ita21
    return interactions


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
                agent.motion_states[ts].x = (agent.motion_states[ts].x - x_start)
                agent.motion_states[ts].y = (agent.motion_states[ts].y - y_start)
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
