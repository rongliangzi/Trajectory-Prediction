import openpyxl
import math
from align_ref_img import counterclockwise_rotate
import numpy as np
from utils import map_vis_without_lanelet
import matplotlib.pyplot as plt
from utils.roundabout_utils import nearest_c_point, on_press
from utils.intersection_utils import cal_dis
import re
import glob
import os
from utils import dataset_reader
from utils import dict_utils
from utils.coordinate_transform import get_frenet
y_range = {19: [1050, 1025], 20: [1026, 1050]}
x_range = {15: [1100, 1077], 16: [1083, 1100], 17: [1020, 1095], 18: [1100, 1020]}
circle_range = {21: [(1054, 1014), (1066.5, 1025.5)], 22: [(1055, 1008), (1072, 1027)],
                23: [(1070.5, 1026.5), (1078, 1017)], 24: [(1066, 1025), (1083, 1007)], }
it_paths = ['17-22-20', '19-21-18', '19-24-17', '15-23-20', '18', '17']
path_mapping = {'17-22-20': '12-16', '19-21-18': '17-11', '19-24-17': '17-13', '15-23-20': '15-16',
                '18': '14-11', '17': '12-13'}


def read_funcs(func_file, wp_n=200):
    map_dir = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/'
    map_name = "DR_USA_Roundabout_EP.osm"
    map_file = map_dir + map_name
    div_path_points = dict()

    with open(func_file) as f:
        lines = f.readlines()
        for line_id, line in enumerate(lines):
            if line_id in y_range.keys():  #
                fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
                map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
                # x=f(y)
                coef = []
                p2 = re.compile('\((.*?)\)', re.S)
                a = re.findall(p2, line)
                for c in a[:-2]:
                    coef.append(eval(c))
                p = np.poly1d(coef)
                yp = np.arange(y_range[line_id][0], y_range[line_id][1], (y_range[line_id][1]-y_range[line_id][0])/wp_n)
                xp = p(yp)
                if line_id == 19:
                    xp -= 0.1
                plt.text(xp[0], yp[0], 'start', fontsize=20)
                plt.text(xp[-1], yp[-1], 'end', fontsize=20)
                plt.plot(xp, yp, linewidth=2)
                plt.savefig('D:/Dev/UCB task/path_imgs/EP/div_{}.png'.format(line_id))
                plt.close()
                div_path_points[line_id] = np.array([[x, y] for x, y in zip(xp, yp)])
            elif line_id in x_range.keys():  #
                fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
                map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
                # y=
                coef = []
                p2 = re.compile(r'[(](.*?)[)]', re.S)
                a = re.findall(p2, line)
                for c in a[:-2]:
                    coef.append(eval(c))
                p = np.poly1d(coef)
                xp = np.arange(x_range[line_id][0], x_range[line_id][1], (x_range[line_id][1]-x_range[line_id][0])/wp_n)
                yp = p(xp)
                if line_id == 18:
                    yp += 0.15
                plt.text(xp[0], yp[0], 'start', fontsize=20)
                plt.text(xp[-1], yp[-1], 'end', fontsize=20)
                plt.plot(xp, yp, linewidth=2)
                plt.savefig('D:/Dev/UCB task/path_imgs/EP/div_{}.png'.format(line_id))
                plt.close()
                div_path_points[line_id] = np.array([[x, y] for x, y in zip(xp, yp)])
            elif line_id in circle_range.keys():
                fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
                map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
                # circle
                p1 = re.compile('\(x-(.*?)\)', re.S)
                a = re.findall(p1, line)
                circle_x = eval(a[0])
                p2 = re.compile('\(y-(.*?)\)', re.S)
                a = re.findall(p2, line)
                circle_y = eval(a[0])
                p2 = re.compile('=(.*)\^', re.S)
                a = re.findall(p2, line)
                circle_r = eval(a[0])
                points_x = [circle_x - circle_r]
                points_y = [circle_y]
                for i in range(359):
                    nx, ny = counterclockwise_rotate(circle_x - circle_r, circle_y, (circle_x, circle_y),
                                                     (i + 1) * math.pi / 180)
                    points_x.append(nx)
                    points_y.append(ny)
                cp_s = circle_range[line_id][0]
                cp_e = circle_range[line_id][1]
                s_i, _, _ = nearest_c_point(cp_s, points_x, points_y)
                e_i, _, _ = nearest_c_point(cp_e, points_x, points_y)
                if e_i < s_i:
                    xp = points_x[s_i:] + points_x[:e_i]
                    yp = points_y[s_i:] + points_y[:e_i]
                else:
                    xp = points_x[s_i: e_i]
                    yp = points_y[s_i: e_i]
                # xp = [cp_s[0]] + xp + [cp_e[0]]
                # yp = [cp_s[1]] + yp + [cp_e[1]]
                plt.plot(xp, yp, linewidth=2)
                plt.text(xp[0], yp[0], 'start', fontsize=20)
                plt.text(xp[-1], yp[-1], 'end', fontsize=20)
                plt.plot(xp, yp, linewidth=2, zorder=30)
                plt.savefig('D:/Dev/UCB task/path_imgs/EP/div_{}.png'.format(line_id))
                plt.close()
                if line_id in [21, 23]:
                    xp = xp[::-1]
                    yp = yp[::-1]
                div_path_points[line_id] = np.array([[x, y] for x, y in zip(xp, yp)])
        ref_path_points = save_it_path(div_path_points, map_file)
    ref_path_id_points = dict()
    for k in ref_path_points.keys():
        ref_path_id_points[path_mapping[k]] = ref_path_points[k]
    return ref_path_id_points


def save_it_path(div_path_points, map_file):
    ref_path_points = dict()
    for ref_path in it_paths:
        # if ref_path != '19-24-17':
        #     continue
        fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
        map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)

        div_paths = ref_path.split('-')
        if len(div_paths) == 1:
            ref_path_points[ref_path] = div_path_points[int(div_paths[0])].copy()
        else:
            ref_path_data0 = div_path_points[int(div_paths[0])].copy()
            ref_path_data1 = div_path_points[int(div_paths[1])].copy()
            ref_path_data2 = div_path_points[int(div_paths[2])].copy()

            dis1 = cal_dis(ref_path_data0, ref_path_data1)
            dis2 = cal_dis(ref_path_data1, ref_path_data2)
            min_id1 = np.argmin(dis1)
            i1, j1 = min_id1 // dis1.shape[1], min_id1 % dis1.shape[1]
            min_id2 = np.argmin(dis2)
            i2, j2 = min_id2 // dis2.shape[1], min_id2 % dis2.shape[1]

            ref_path_data0 = ref_path_data0[:i1]
            ref_path_data1 = ref_path_data1[j1:i2 + 1]
            ref_path_data2 = ref_path_data2[j2:]
            if ref_path == '17-22-20':
                ref_path_data0[:, 1] += 0.235
                # ref_path_data1[-1] = (ref_path_data1[-2]+ref_path_data2[2])/2
                ref_path_data2 = ref_path_data2[1:]
                ref_path_data2[:, 0] += 0.01
            elif ref_path == '19-21-18':
                # ref_path_data1[-1] = (ref_path_data1[-2] + ref_path_data2[1]) / 2
                # ref_path_data2 = ref_path_data2[1:]
                ref_path_data0[:, 0] -= 0.04
                ref_path_data2[:, 1] += 0.01
                pass
            elif ref_path == '19-24-17':
                ref_path_data0[:, 0] -= 0.1
                ref_path_data1[0] = (ref_path_data0[-1]+ref_path_data1[1])/2
                ref_path_data2 = ref_path_data2[1:]
                ref_path_data2[:, 1] += 0.483

            elif ref_path == '15-23-20':
                ref_path_data1[-2] = (ref_path_data1[-4] + ref_path_data2[0])/2
                ref_path_data1[-1] = (ref_path_data1[-2] + ref_path_data2[0])/2
                ref_path_data1[-3] = (ref_path_data1[-2] + ref_path_data1[-4])/2
            ref_path_points[ref_path] = np.vstack((ref_path_data0, ref_path_data1, ref_path_data2))
        xp, yp = [point[0] for point in ref_path_points[ref_path]], [point[1] for point in ref_path_points[ref_path]]
        xy_p = np.array([[x, y] for x, y in zip(xp, yp)])
        plt.text(xp[0], yp[0], 'start', fontsize=20)
        plt.text(xp[-1], yp[-1], 'end', fontsize=20)
        plt.plot(xp, yp, linewidth=1, zorder=30, marker='x')
        plt.title(ref_path)
        # fig.canvas.mpl_connect('button_press_event', on_press)
        # plt.show()
        plt.savefig('D:/Dev/UCB task/path_imgs/EP/{}.png'.format(ref_path))
        plt.close()
        ref_path_points[ref_path] = xy_p
    return ref_path_points


def visualize_ref_path(way_points, raw, map_file):
    fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
    map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
    xp, yp = [point[0] for point in way_points], [point[1] for point in way_points]

    plt.text(xp[0], yp[0], 'start', fontsize=20)
    plt.text(xp[-1], yp[-1], 'end', fontsize=20)
    plt.plot(xp, yp, linewidth=1, zorder=30, marker='x', c='r')
    plt.plot([point[0] for point in raw], [point[1] for point in raw], linewidth=1, zorder=30, marker='x', c='g')
    fig.canvas.mpl_connect('button_press_event', on_press)
    plt.show()


def get_track_label(dir_name, ref_path_points, ref_frenet):
    print('saving each agent\'s info')
    csv_dict = dict()
    # collect data to construct a dict from all csv
    paths = glob.glob(os.path.join(dir_name, '*.csv'))
    paths.sort()
    for csv_name in paths:
        print(csv_name)
        track_dictionary = dataset_reader.read_tracks(csv_name)
        tracks = dict_utils.get_value_list(track_dictionary)
        csv_agents = dict()
        for agent in tracks:
            start_area = agent.start_area
            end_area = agent.end_area
            if start_area == 'NAN' or end_area == 'NAN':
                continue
            path_name = str(start_area) + '-' + str(end_area)
            if path_name not in ref_path_points.keys():
                continue
            xy_points = ref_path_points[path_name]
            start_ts = -1
            for ts in range(agent.time_stamp_ms_first, agent.time_stamp_ms_last + 100, 100):
                ms = agent.motion_states[ts]
                x = ms.x
                y = ms.y
                _, _, _, drop_head, _ = get_frenet(x, y, xy_points, ref_frenet[path_name])
                if drop_head == 0:
                    start_ts = ts
                    break
            end_ts = -1
            for ts in range(agent.time_stamp_ms_last, agent.time_stamp_ms_first - 100, -100):
                ms = agent.motion_states[ts]
                x = ms.x
                y = ms.y
                _, _, _, _, drop_tail = get_frenet(x, y, xy_points, ref_frenet[path_name])
                if drop_tail == 0:
                    end_ts = ts - 100
                    break
            if start_ts == -1:
                print('start_ts:-1', csv_name, agent.track_id, path_name)
            if end_ts == -1:
                print('end ts: -1', csv_name, agent.track_id, path_name)
            agent.time_stamp_ms_first = start_ts
            agent.time_stamp_ms_last = end_ts
            # calculate frenet s,d and velocity along s direction
            max_min_dis_traj = 0
            for ts in range(agent.time_stamp_ms_first, agent.time_stamp_ms_last + 100, 100):
                x = agent.motion_states[ts].x
                y = agent.motion_states[ts].y
                dis = cal_dis(np.array([[x, y]]), xy_points)
                min_dis = dis.min()
                max_min_dis_traj = max(min_dis, max_min_dis_traj)
                f_s, f_d, proj, _, _ = get_frenet(x, y, xy_points, ref_frenet[path_name])
                agent.motion_states[ts].frenet_s = f_s
                agent.motion_states[ts].frenet_d = f_d
                agent.motion_states[ts].proj = proj
                if ts > agent.time_stamp_ms_first:
                    vs = (f_s - agent.motion_states[ts - 100].frenet_s) / 0.1
                    agent.motion_states[ts].vs = vs
            if max_min_dis_traj > 100:
                continue
            agent.motion_states[agent.time_stamp_ms_first].vs = agent.motion_states[agent.time_stamp_ms_first + 100].vs
            agent_dict = dict()
            agent_dict['track_id'] = agent.track_id
            agent_dict['time_stamp_ms_first'] = agent.time_stamp_ms_first
            agent_dict['time_stamp_ms_last'] = agent.time_stamp_ms_last
            agent_dict['ref path'] = path_name
            agent_dict['motion_states'] = dict()
            for ts in range(agent.time_stamp_ms_first, agent.time_stamp_ms_last + 100, 100):
                ms = agent.motion_states[ts]
                agent_dict['motion_states'][ts] = dict()
                agent_dict['motion_states'][ts]['time_stamp_ms'] = ms.time_stamp_ms
                agent_dict['motion_states'][ts]['x'] = ms.x
                agent_dict['motion_states'][ts]['y'] = ms.y
                agent_dict['motion_states'][ts]['vx'] = ms.vx
                agent_dict['motion_states'][ts]['vy'] = ms.vy
                agent_dict['motion_states'][ts]['psi_rad'] = ms.psi_rad
                agent_dict['motion_states'][ts]['vs'] = ms.vs
                agent_dict['motion_states'][ts]['frenet_s'] = ms.frenet_s
                agent_dict['motion_states'][ts]['frenet_d'] = ms.frenet_d
                agent_dict['motion_states'][ts]['proj'] = ms.proj
            csv_agents[agent.track_id] = agent_dict
        csv_dict[csv_name[-7:-4]] = csv_agents
    return csv_dict
