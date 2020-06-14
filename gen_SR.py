import matplotlib.patches as patches
import scipy.io as scio
import numpy as np
import math
import glob
import os
import pickle
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from utils import map_vis_without_lanelet
from align_ref_img import counterclockwise_rotate
from utils.new_coor_ref_path_utils import judge_in_box
from utils import dataset_reader
from utils import dict_utils
from utils.coordinate_transform import get_frenet

SR_starting_area_dict = dict()
SR_starting_area_dict[1] = dict()
SR_starting_area_dict[1]['x'] = [957.2, 955.2, 968.8, 968.7]
SR_starting_area_dict[1]['y'] = [1015.8, 1009.6, 1006.7, 1013.6]
SR_starting_area_dict[3] = dict()
SR_starting_area_dict[3]['x'] = [986.6, 980.8, 989.6, 993.2]
SR_starting_area_dict[3]['y'] = [1058, 1047, 1046.8, 1059.5]
SR_starting_area_dict[5] = dict()
SR_starting_area_dict[5]['x'] = [1015.3, 1013.2, 1027, 1029.3]
SR_starting_area_dict[5]['y'] = [1034.9, 1027, 1024, 1030]
SR_starting_area_dict[7] = dict()
SR_starting_area_dict[7]['x'] = [989, 988.2, 993.4, 997.2]
SR_starting_area_dict[7]['y'] = [991.5, 984.9, 984.8, 991.1]

SR_end_area_dict = dict()
SR_end_area_dict[2] = dict()
SR_end_area_dict[2]['x'] = [953.2, 954, 968.7, 968]
SR_end_area_dict[2]['y'] = [1032.6, 1025.7, 1027.8, 1037]
SR_end_area_dict[4] = dict()
SR_end_area_dict[4]['x'] = [994.5, 995, 1004.4, 1001.2]
SR_end_area_dict[4]['y'] = [1056, 1047, 1046, 1056.6]
SR_end_area_dict[6] = dict()
SR_end_area_dict[6]['x'] = [1013.1, 1014, 1025.1, 1025.4]
SR_end_area_dict[6]['y'] = [1013.3, 1004, 1006.3, 1012.8]
SR_end_area_dict[8] = dict()
SR_end_area_dict[8]['x'] = [981.2, 983.9, 987.8, 987.9]
SR_end_area_dict[8]['y'] = [995.2, 989, 988.5, 995.4]


def on_press(event):
    print("my position:", event.button, event.xdata, event.ydata)


def judge_start(track):
    # judge if in some starting area frame by frame
    for ts in range(track.time_stamp_ms_first, track.time_stamp_ms_last+100, 100):
        motion_state = track.motion_states[ts]
        cur_p = (motion_state.x, motion_state.y)
        for k, v in SR_starting_area_dict.items():
            in_box = judge_in_box(v['x'], v['y'], cur_p)
            if in_box == 1:
                return k
    return 0


def judge_end(track):
    # judge if in some starting area frame by frame
    for ts in range(track.time_stamp_ms_first, track.time_stamp_ms_last+100, 100):
        motion_state = track.motion_states[ts]
        cur_p = (motion_state.x, motion_state.y)
        for k, v in SR_end_area_dict.items():
            in_box = judge_in_box(v['x'], v['y'], cur_p)
            if in_box == 1:
                return k
    return 0


# transform points in a ref path to frenet coordinate
def ref_point2frenet(x_points, y_points):
    points_frenet_s = [0]
    for i in range(1, len(x_points)):
        prev_x = x_points[i-1]
        prev_y = y_points[i-1]
        x = x_points[i]
        y = y_points[i]
        s_dis = ((x-prev_x)**2+(y-prev_y)**2)**0.5
        points_frenet_s.append(points_frenet_s[i-1]+s_dis)
    frenet_s_points = np.array(points_frenet_s)
    return frenet_s_points


def ref_paths2frenet(ref_paths):
    ref_frenet_coor = dict()
    for path_name, path_points in ref_paths.items():
        points_frenet_s = ref_point2frenet(path_points[:, 0], path_points[:, 1])
        ref_frenet_coor[path_name] = points_frenet_s
    return ref_frenet_coor


def get_track_label(dir_name):
    csv_dict = dict()
    # collect data to construct a dict from all csv
    paths = glob.glob(os.path.join(dir_name, '*.csv'))
    paths.sort()
    for csv_name in paths:
        print(csv_name)
        track_dictionary = dataset_reader.read_tracks(csv_name)
        tracks = dict_utils.get_value_list(track_dictionary)
        agent_path = dict()
        for agent in tracks:
            # # transform the coordinate
            # for ts in range(agent.time_stamp_ms_first, agent.time_stamp_ms_last+100, 100):
            #     agent.motion_states[ts].x = (agent.motion_states[ts].x - x_s) * rate
            #     agent.motion_states[ts].y = (agent.motion_states[ts].y - y_s) * rate
            start_area = judge_start(agent)
            end_area = judge_end(agent)
            if start_area == 0 or end_area == 0:
                print(0)
                continue
            k = str(start_area) + '-' + str(end_area)
            if k not in ref_path_points:
                k = str(start_area) + '--1-' + str(end_area)
            xy_points = ref_path_points[k]
            agent_frenet = []
            for ts in range(agent.time_stamp_ms_first, agent.time_stamp_ms_last + 100, 100):
                x = agent.motion_states[ts].x
                y = agent.motion_states[ts].y
                psi_rad = agent.motion_states[ts].psi_rad
                frenet_s, _ = get_frenet(x, y, psi_rad, xy_points, ref_point_frenet[k])
                agent_frenet.append(frenet_s)
            agent_path[agent.track_id] = {'data': agent, 'ref path': k,
                                          'frenet': agent_frenet}
        csv_dict[csv_name[-7:-4]] = agent_path
    return csv_dict


def plot_raw_ref_path(map_file, all_points, circle_point):
    fig, axes = plt.subplots(1, 1, figsize=(30, 20), dpi=100)
    map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
    for way_points in all_points[0, :]:
        x = [p[0] for p in way_points]
        y = [p[1] for p in way_points]
        plt.plot(x, y, linewidth=4)
    for key, v in SR_starting_area_dict.items():
        x = v['x']
        y = v['y']
        plt.text(x[0], y[0], key, fontsize=20)
        plt.plot(x[0:2], y[0:2], c='r', zorder=40)
        plt.plot(x[1:3], y[1:3], c='r', zorder=40)
        plt.plot(x[2:4], y[2:4], c='r', zorder=40)
        plt.plot(x[3:] + x[0:1], y[3:] + y[0:1], c='r', zorder=40)
    for key, v in SR_end_area_dict.items():
        x = v['x']
        y = v['y']
        plt.text(x[0], y[0], key, fontsize=20)
        plt.plot(x[0:2], y[0:2], c='r', zorder=40)
        plt.plot(x[1:3], y[1:3], c='r', zorder=40)
        plt.plot(x[2:4], y[2:4], c='r', zorder=40)
        plt.plot(x[3:] + x[0:1], y[3:] + y[0:1], c='r', zorder=40)
    for p in circle_point:
        if math.isnan(p[0][0]):
            continue
        circle = patches.Circle(p[0], 1, color='r', zorder=3)
        axes.add_patch(circle)
    fig.canvas.mpl_connect('button_press_event', on_press)
    plt.show()


def plot_ref_path_divided(map_file, ref_path_points):
    fig, axes = plt.subplots(2, 3)
    start = dict()
    i = -1
    keys = sorted(ref_path_points.keys())
    for k in keys:
        v = ref_path_points[k]
        st = k.split('-')[0]
        if st not in start:
            i += 1
            start[st] = 1
            map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes[i//3][i % 3], 0, 0)
        xp = [p[0] for p in v]
        yp = [p[1] for p in v]
        axes[i // 3][i % 3].plot(xp, yp, linewidth=4)
    plt.subplots_adjust(top=1, bottom=0, left=0.05, right=0.95, hspace=0.1, wspace=0.1)
    fig.canvas.mpl_connect('button_press_event', on_press)
    plt.show()


def plot_ref_path(map_file, ref_path_points):
    fig, axes = plt.subplots(1, 1)
    map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
    keys = sorted(ref_path_points.keys())
    for k in keys:
        v = ref_path_points[k]
        xp = [p[0] for p in v]
        yp = [p[1] for p in v]
        plt.plot(xp, yp, linewidth=4)
    fig.canvas.mpl_connect('button_press_event', on_press)
    plt.show()


def residuals(p, x, y):
    a, b, r = p
    return r**2 - (y - b) ** 2 - (x - a) ** 2


def fit_circle(circle_point):
    circle_x = []
    circle_y = []
    for p in circle_point:
        if math.isnan(p[0][0]):
            continue
        circle_x.append(p[0][0])
        circle_y.append(p[0][1])
    x = np.array(circle_x)
    y = np.array(circle_y)
    result = leastsq(residuals, np.array([1, 1, 1]), args=(x, y))
    a, b, r = result[0]
    r = max(-r, r)
    print("a=", a, "b=", b, "r=", r)
    cx = [a-r]
    cy = [b]
    for i in range(359):
        nx, ny = counterclockwise_rotate(a-r, b, (a, b), (i+1)*math.pi/180)
        cx.append(nx)
        cy.append(ny)
    return cx, cy


def nearest_c_point(p, cx, cy):
    min_dis = 1e8
    xn, yn = 0, 0
    min_i = 0
    for i in range(len(cx)):
        x, y = cx[i], cy[i]
        if (x-p[0])**2 + (y-p[1])**2 < min_dis:
            min_dis = (x-p[0])**2 + (y-p[1])**2
            xn, yn = x, y
            min_i = i
    return min_i, xn, yn


def get_ref_path(data, cx, cy):
    ref_path_points = dict()
    para_path = data['para_path'][0]
    circle_merge_point = data['circle_merge_point'][0]
    branchID = data['branchID'][0]
    pre = dict()
    post = dict()

    for i in range(len(branchID)):
        s = branchID[i][0]
        if s[-1] == -1:
            min_i, _, _ = nearest_c_point(circle_merge_point[i][0], cx, cy)
            d = {'min_i': min_i, 'path': para_path[i]}
            pre[str(s[0])] = d
        elif s[0] == -1:
            min_i, _, _ = nearest_c_point(circle_merge_point[i][0], cx, cy)
            d = {'min_i': min_i, 'path': para_path[i]}
            post[str(s[1])] = d
        else:
            label = str(s[0]) + '-' + str(s[1])
            ref_path_points[label] = para_path[i]

    for k1, v1 in pre.items():
        for k2, v2 in post.items():
            if k1+'-'+k2 in ref_path_points.keys():
                continue
            label = k1+'--1-'+k2
            i1 = v1['min_i']
            i2 = v2['min_i']
            if i2 > i1:
                cpx = cx[i1:i2+1]
                cpy = cy[i1:i2+1]
            else:
                cpx = cx[i1:] + cx[:i2+1]
                cpy = cy[i1:] + cy[:i2+1]
            cp = np.array([[x, y] for x, y in zip(cpx, cpy)])
            ref_path_points[label] = np.vstack((v1['path'], cp, v2['path']))
    return ref_path_points


if __name__ == '__main__':
    map_dir = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/'
    map_name = "DR_USA_Roundabout_SR.osm"
    dataFile = 'D:/Dev/UCB task/Segmented_reference_path_DR_USA_Roundabout_SR.mat'
    data = scio.loadmat(dataFile)
    para_path = data['Segmented_reference_path']['para_path']
    circle_merge_point = data['Segmented_reference_path']['circle_merge_point'][0]
    plot_raw_ref_path(map_dir + map_name, para_path, circle_merge_point)

    cx, cy = fit_circle(circle_merge_point)
    # a dict, call by path return an array(x,2)
    ref_path_points = get_ref_path(data['Segmented_reference_path'], cx, cy)

    # plot_ref_path_divided(map_dir + map_name, ref_path_points)
    # plot_ref_path(map_dir + map_name, ref_path_points)

    # a dict, call by path return an array(x,1): frenet of ref path points
    ref_point_frenet = ref_paths2frenet(ref_path_points)
    csv_data = get_track_label('D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/DR_USA_Roundabout_SR/')
    pickle_file = open('D:/Dev/UCB task/pickle/track_path_frenet_SR.pkl', 'wb')
    pickle.dump(csv_data, pickle_file)
    pickle_file.close()

