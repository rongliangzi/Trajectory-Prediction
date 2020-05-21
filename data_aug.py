from utils import dataset_reader
from utils import dict_utils
from utils import map_vis_without_lanelet
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
import glob
import os
import pickle

starting_area_dict = dict()
starting_area_dict[2] = dict()
starting_area_dict[2]['x'] = [992, 992.2, 1002.3, 1002]
starting_area_dict[2]['y'] = [1005.5, 1002.5, 1002, 1005.5]
starting_area_dict[2]['stoplinex'] = [1000.1, 1000.2]
starting_area_dict[2]['stopliney'] = [1005.5, 1002.5]
starting_area_dict[3] = dict()
starting_area_dict[3]['x'] = [992, 992.2, 1002.3, 1002]
starting_area_dict[3]['y'] = [1002.5, 997.5, 993, 1002]
starting_area_dict[3]['stoplinex'] = [1000.15, 1000.3]
starting_area_dict[3]['stopliney'] = [1001.5, 997]
starting_area_dict[6] = dict()
starting_area_dict[6]['x'] = [1024, 1025.5, 1028.9, 1027.6]
starting_area_dict[6]['y'] = [996.5, 986, 986.5, 997]
starting_area_dict[6]['stoplinex'] = [1024, 1028]
starting_area_dict[6]['stopliney'] = [994.6, 994.8]
starting_area_dict[7] = dict()
starting_area_dict[7]['x'] = [1027.9, 1029.2, 1036.8, 1038.3]
starting_area_dict[7]['y'] = [997.5, 986.5, 987, 998]
starting_area_dict[7]['stoplinex'] = [1028.8, 1036]
starting_area_dict[7]['stopliney'] = [994.8, 995]
starting_area_dict[9] = dict()
starting_area_dict[9]['x'] = [1031.5, 1032.5, 1042.5, 1042.3]
starting_area_dict[9]['y'] = [1017, 1005.5, 1006, 1010]
starting_area_dict[9]['stoplinex'] = [1033.3, 1034.3]
starting_area_dict[9]['stopliney'] = [1014, 1006]
starting_area_dict[12] = dict()
starting_area_dict[12]['x'] = [1015.3, 1015.5, 1020.5, 1019]
starting_area_dict[12]['y'] = [1020, 1010.5, 1010.4, 1020.2]
starting_area_dict[12]['stoplinex'] = [1016.5, 1020]
starting_area_dict[12]['stopliney'] = [1012, 1012]
starting_area_dict[13] = dict()
starting_area_dict[13]['x'] = [1011.5, 1011.7, 1016, 1015.5]
starting_area_dict[13]['y'] = [1020, 1010.5, 1010.4, 1020.2]
starting_area_dict[13]['stoplinex'] = [1011.7, 1016.3]
starting_area_dict[13]['stopliney'] = [1012, 1012]
starting_area_dict[14] = dict()
starting_area_dict[14]['x'] = [1006, 996.2, 1011.4, 1011.1]
starting_area_dict[14]['y'] = [1020, 1010, 1009.9, 1020.2]
starting_area_dict[14]['stoplinex'] = [1004, 1011.3]
starting_area_dict[14]['stopliney'] = [1012, 1012]


def judge_start_v1(ms, track):
    if ms.x < 1007 and 1002.6 < ms.y < 1006 and ms.vx > 0:
        return 2
    elif ms.x < 1007 and 997 < ms.y < 1002.6 and ms.vx > 0:
        return 3
    elif ms.y < 980 and 1025 < ms.x < 1030.5 and ms.vy > 0:
        return 6
    elif ms.y < 980 and 1030.5 < ms.x < 1037 and ms.vy > 0:
        return 7
    elif ms.x > 1039 and ms.y > 1007 and ms.vx <= 0:
        return 9
    elif ms.y > 1017 and ms.vy <= 0:
        for ts in range(track.time_stamp_ms_first, track.time_stamp_ms_last, 100):
            motion_state = track.motion_states[ts]
            if motion_state.x < 1011 and 1013 < motion_state.y < 1025:
                return 14
            elif 1011 < motion_state.x < 1016 and 1013 < motion_state.y < 1025:
                return 13
            elif 1016 < motion_state.x < 1020 and 1013 < motion_state.y < 1025:
                return 12
        return 0
    else:
        return 0


def judge_start(ms, track):
    # judge if in some starting area frame by frame
    for ts in range(track.time_stamp_ms_first, track.time_stamp_ms_last, 100):
        motion_state = track.motion_states[ts]
        cur_p = (motion_state.x, motion_state.y)
        for k, v in starting_area_dict.items():
            in_box = judge_in_box(v['x'], v['y'], cur_p)
            if in_box == 1:
                return k
    return 0


def judge_end(ms, track):
    if ms.x < 1000 and abs(ms.y-1007) < 3 and ms.vx < -0.02:
        return 1
    elif ms.y < 980 and ms.vy < 0:
        if ms.x < 1007:
            return 15
        elif ms.x < 1013:
            return 4
        elif 1013 < ms.x:
            return 5
    elif ms.x > 1045 and ms.vx > 0:
        return 8
    elif ms.y > 1020 and ms.x > 1022 and ms.vy > 0:
        return 10
    elif ms.y > 1020 and ms.x <= 1022 and ms.vy > 0:
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
        y_mask = y_back > 965
        x_mask = x_back < 1080
        mask = (x_mask.astype(np.int) + y_mask.astype(np.int)) > 1
    elif k == '12-8':
        mask = (y_back < 1035) & (x_back < 1085)
    elif k == '9-5':
        mask = (y_back > 968) & (y_back < 1035) & (x_back > 970) & (x_back < 1085)
    else:
        mask = (y_back > 965) & (y_back < 1035) & (x_back > 970) & (x_back < 1085)
    y_back = y_back[mask]
    x_back = x_back[mask]
    path_w = np.array(path_w)
    avg_path_w = np.average(path_w)

    # reverse the direction
    if k in ['6-1', '7-10', '9-1', '9-4', '9-5', '9-10', '6-4', '7-11', '6-10', '6-11', '9-11', '13-4']:
        x_back = x_back[::-1]
        y_back = y_back[::-1]

    poly_func = ''
    return [x_back, y_back, x, y, avg_path_w], poly_func


def get_ref_paths(base_path, dir_name):
    '''
    :param base_path:
    :param dir_name:
    :param csv_num:
    :return: all ref paths in a dict. each item includes [x, y, raw x, raw y, path w], poly func in str
    '''
    trajectory = dict()
    csv_dict = dict()
    f = open('D:/Dev/UCB task/poly.txt', 'r')
    lines = f.readlines()
    ref_index_dict = dict()
    for line in lines[1:]:
        sp = line.split(',')
        ref_index_dict[sp[1]+'-'+sp[2]] = sp[0]
    # collect data to construct a dict from all csv
    paths = glob.glob(os.path.join(base_path + dir_name, '*.csv'))
    paths.sort()
    for csv_name in paths:
        track_dictionary = dataset_reader.read_tracks(csv_name)
        tracks = dict_utils.get_value_list(track_dictionary)
        agent_path = list()
        for agent in tracks:
            first_ms = agent.motion_states[agent.time_stamp_ms_first]
            last_ms = agent.motion_states[agent.time_stamp_ms_last]
            start_area = judge_start(first_ms, agent)
            end_area = judge_end(last_ms, agent)
            if start_area == 0 or end_area == 0:
                pass
            else:
                k = str(start_area) + '-' + str(end_area)
                agent_path.append([agent, k])
                if k not in trajectory:
                    trajectory[k] = [agent]
                elif k not in ['12-10', '13-10']:
                    trajectory[k].append(agent)
        csv_dict[csv_name[-7:-4]] = agent_path
    ref_paths = dict()
    beta_dict = {'7-8': math.pi / 4, '12-8': -math.pi / 6, '2-10': math.pi / 4, '2-11': math.pi / 4,
                 '6-1': -math.pi / 6, '3-4': -math.pi / 6, '9-4': math.pi / 4, '9-5': math.pi / 4,
                 '9-10': -math.pi / 6, '13-5': -math.pi / 6, '14-4': -math.pi / 6, '6-11': -math.pi / 6,
                 '7-10': 0, '2-8': 0, '3-8': 0, '9-1': 0, '12-5': math.pi / 5, '13-8': -math.pi / 6,
                 '14-1': math.pi / 4, '14-15': -math.pi / 6, '13-4': math.pi / 5, '14-5': math.pi / 4,
                 '12-10': math.pi / 12, '7-11': -math.pi / 6, '9-11': -math.pi / 6, '13-10': math.pi / 12,
                 '2-4': -math.pi / 6, '3-5': -math.pi / 6, '6-10': -math.pi / 6, '6-4': 0}
    poly_dict = {'7-8': 6, '12-8': 6, '2-10': 6, '2-11': 6, '6-1': 6, '3-4': 6, '9-4': 6, '9-5': 6,
                 '9-10': 4, '13-5': 1, '14-4': 1, '6-11': 3, '7-10': 1, '2-8': 2, '3-8': 2, '9-1': 4,
                 '12-5': 3, '13-8': 6, '14-1': 5, '14-15': 1, '13-4': 3, '14-5': 6, '12-10': 4, '7-11': 4,
                 '9-11': 4, '13-10': 6, '2-4': 4, '3-5': 4, '6-10': 4, '6-4': 6}
    # for trajectories in one ref path
    rare_paths = []
    for ref_i, (k, v) in enumerate(sorted(trajectory.items())):
        xy = []
        path_w = []
        # save all (x,y) points in xy
        for track in v:
            path_w.append(track.width)
            for ts in range(track.time_stamp_ms_first, track.time_stamp_ms_last, 100):
                motion_state = track.motion_states[ts]
                if k == '12-8' and motion_state.x < 1015:
                    pass
                elif k in ['7-8', '9-5', '9-3'] and motion_state.y < 970:
                    # add some data points
                    for i in range(20):
                        xy.append([motion_state.x + random.random() * 0.5,
                                   motion_state.y + (random.random() - 0.5) * 0.4])
                        xy.append([motion_state.x + random.random() * 0.5,
                                   motion_state.y + random.random() - 10])
                elif k == '12-8' and motion_state.x > 1075:
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


def cal_dis(x, y):
    # calculate distance matrix for [x1,xm] and [y1,yn]
    row_x, col_x = x.shape
    row_y, col_y = y.shape
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (row_x, 1)), repeats=row_y, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (row_y, 1)), repeats=row_x, axis=1).T
    dis = x2 + y2 - 2 * xy
    return dis


def find_crossing_point(dis, x, y):
    pos = np.argmin(dis)
    xid = pos // y.shape[0]
    yid = pos % y.shape[0]
    # the shortest distance is from xid-th point in x to yid-th point in y
    crossing_point = (x[xid]+y[yid])/2
    # print('({:.3f},{:.3f})'.format(crossing_point[0], crossing_point[1]))
    return crossing_point, xid, yid


def find_merging_point(x, y, dis, th=0.8):
    # minimum distance of each row
    min_dis = dis.min(axis=1)
    # id of y for minimum distance of each row
    y_id = np.argmin(dis, axis=1)
    assert len(min_dis) > 5
    # try to find a merging point. two points before it > th and two points behind it < th
    # if not found, return None
    for i in range(2, len(min_dis)-2):
        if min_dis[i-1] > th > min_dis[i+1] and min_dis[i-2] > th > min_dis[i+2]:
            merging_point = (x[i]+y[y_id[i]])/2
            return merging_point, i, y_id[i]
        return None


def rotate_aug(x, y, intersection, theta):
    # rotate the x,y point counterclockwise at angle theta around intersection point
    x_rot = (x - intersection[0]) * math.cos(theta) - (y - intersection[1]) * math.sin(theta) + intersection[0]
    y_rot = (x - intersection[0]) * math.sin(theta) + (y - intersection[1]) * math.cos(theta) + intersection[1]
    return x_rot, y_rot


def plot_intersection(x1, y1, w1, x2, y2, w2, fig_name, intersection, id1, id2):
    '''
    :param x1: x list
    :param y1: y list
    :param w1: line width
    :param x2:
    :param y2:
    :param w2:
    :param fig_name: figure name to save
    :param intersection: (x,y) of intersection
    :param id1: id of the intersection point in x1
    :param id2:
    :return: save the figure
    '''
    d = 4  # fig size
    r = 60  # range of x and y
    fig, axes = plt.subplots(1, 1, figsize=(d, d), dpi=100)
    # set bg to black
    axes.patch.set_facecolor("k")

    # calculate the k as the tangent
    if id1+1 >= len(x1):
        delta_y1, delta_x1 = y1[id1] - y1[id1 - 1], x1[id1] - x1[id1 - 1]
    elif id1-1 < 0:
        delta_y1, delta_x1 = y1[id1 + 1] - y1[id1], x1[id1 + 1] - x1[id1]
    else:
        delta_y1, delta_x1 = y1[id1 + 1] - y1[id1 - 1], x1[id1 + 1] - x1[id1 - 1]
    k1 = delta_y1 / delta_x1
    theta1 = math.atan(k1)
    # convert from -pi/2~pi/2 to 0~pi
    if theta1 < 0:
        theta1 += math.pi
    # convert to pi~2pi if needed
    if delta_x1 < 0 < k1 or k1 < 0 < delta_x1:
        theta1 += math.pi
    if id2+1 >= len(x2):
        delta_y2, delta_x2 = (y2[id2] - y2[id2 - 1]), (x2[id2] - x2[id2 - 1])
    elif id2-1 < 0:
        delta_y2, delta_x2 = (y2[id2 + 1] - y2[id2]), (x2[id2 + 1] - x2[id2])
    else:
        delta_y2, delta_x2 = (y2[id2 + 1] - y2[id2 - 1]), (x2[id2 + 1] - x2[id2 - 1])

    k2 = delta_y2 / delta_x2
    theta2 = math.atan(k2)
    # convert from -pi/2~pi/2 to 0~pi
    if theta2 < 0:
        theta2 += math.pi
    # convert to pi~2pi if needed
    if delta_x2 < 0 < k2 or k2 < 0 < delta_x2:
        theta2 += math.pi
    # theta of angle bisector
    avg_theta = (theta1+theta2)/2
    theta1_rot = theta1 - avg_theta
    k1_rot = math.tan(theta1_rot)
    theta2_rot = theta2 - avg_theta
    k2_rot = math.tan(theta2_rot)
    # rotate according to angle bisector to align
    x1_rot, y1_rot = rotate_aug(x1, y1, intersection, -avg_theta)
    x2_rot, y2_rot = rotate_aug(x2, y2, intersection, -avg_theta)
    plt.plot(x1_rot, y1_rot, linewidth=w1 * 72 * d // r, color='b')
    plt.plot(x2_rot, y2_rot, linewidth=w2 * 72 * d // r, color='g')
    # plt.plot(x1, y1, linewidth=w1 * 72 * d // r, color='b')
    # plt.plot(x2, y2, linewidth=w2 * 72 * d // r, color='g')
    circle = patches.Circle(intersection, (w1 + w2) * 2 * d / r, color='r', zorder=3)
    axes.add_patch(circle)
    # draw the arrow whose length is al
    al = 8
    axes.arrow(x1[id1], y1[id1], al/(k1_rot**2+1)**0.5, al * k1_rot/(k1_rot**2+1)**0.5, zorder=4,
               color='purple', width=0.2, head_width=0.6)
    axes.arrow(x2[id2], y2[id2], al/(k2_rot**2+1)**0.5, al * k2_rot/(k2_rot**2+1)**0.5, zorder=5,
               color='yellow', width=0.2, head_width=0.6)
    # axes.arrow(x1[id1], y1[id1], al / (k1 ** 2 + 1) ** 0.5, al * k1 / (k1 ** 2 + 1) ** 0.5, zorder=4,
    #            color='purple', width=0.2, head_width=0.6)
    # axes.arrow(x2[id2], y2[id2], al / (k2 ** 2 + 1) ** 0.5, al * k2 / (k2 ** 2 + 1) ** 0.5, zorder=5,
    #            color='yellow', width=0.2, head_width=0.6)
    # set x y range
    plt.xlim(intersection[0]-r//2, intersection[0]+r//2)
    plt.ylim(intersection[1]-r//2, intersection[1]+r//2)

    # remove the white biankuang
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig('D:/Dev/UCB task/intersection_figs/{}.png'.format(fig_name))
    plt.close()


def find_intersection(seq1, seq2):
    X1, Y1 = seq1[0], seq1[1]
    X2, Y2 = seq2[0], seq2[1]
    x = np.array([[x1, y1] for x1, y1 in zip(X1, Y1)])
    y = np.array([[x2, y2] for x2, y2 in zip(X2, Y2)])
    dis = cal_dis(x, y)
    min_dis = dis.min()

    if min_dis > 4:
        # no intersection point
        return None
    intersection = find_merging_point(x, y, dis)
    # no merging point, find crossing point
    if intersection is None:
        intersection, xid, yid = find_crossing_point(dis, x, y)
    else:
        intersection, xid, yid = intersection
    return intersection, xid, yid


def plot_global_image(ref_paths, fig_name, work_dir, bg=False):
    '''
    :param ref_paths:
    :param fig_name: figure name
    :param work_dir: the dir to save
    :param bg: if plot background
    :return:
    '''
    d = 8  # fig size
    r = 100  # range of x and y
    fig, axes = plt.subplots(1, 1, figsize=(d, d), dpi=200)
    if bg:
        # load and draw the lanelet2 map, either with or without the lanelet2 library
        lanelet_map_file = "D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/DR_USA_Intersection_MA.osm"
        map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, 0, 0)
        fig_name += '_bg'
    else:
        # set bg to black
        axes.patch.set_facecolor("k")
    for path_name in ref_paths.keys():
        x0, y0, _, _, w0 = ref_paths[path_name]
        plt.plot(x0, y0, linewidth=w0 * 36 * d // r, color='b')
    # set x y range
    xs, ys = 980, 950
    plt.xlim(xs, xs + r)
    plt.ylim(ys, ys + r)

    # remove the white biankuang
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    save_dir = work_dir+'global_images/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.savefig(save_dir+'{}.png'.format(fig_name))
    plt.close()
    for path_name in ref_paths.keys():
        d = 8  # fig size
        r = 100  # range of x and y
        fig, axes = plt.subplots(1, 1, figsize=(d, d), dpi=200)
        lanelet_map_file = "D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/DR_USA_Intersection_MA.osm"
        map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, 0, 0)
        x0, y0, x1, y1, w0 = ref_paths[path_name]
        plt.plot(x0, y0, linewidth=w0 * 36 * d // r, color='b', alpha=0.5)
        plt.scatter(x1, y1, s=1)
        xs, ys = 980, 950
        plt.xlim(xs, xs + r)
        plt.ylim(ys, ys + r)
        # remove the white biankuang
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(save_dir+'{}_{}.png'.format(fig_name, path_name))
        plt.close()


def plot_rotate_crop(ref_paths, path_name, path_info, fig_name, theta, start_point, save_dir):
    d = 4  # fig size
    r = 40  # range of x and y
    dpi = 50
    fig, axes = plt.subplots(1, 1, figsize=(d, d), dpi=dpi)
    # set bg to black
    axes.patch.set_facecolor("k")
    x1, y1, w1 = ref_paths[path_name][0], ref_paths[path_name][1], ref_paths[path_name][4]
    x1_rot, y1_rot = rotate_aug(x1, y1, start_point, theta)
    plt.plot(x1_rot, y1_rot, linewidth=w1 * 36 * d // r, color='r', zorder=25)
    for srd_path_name, intersection in path_info.items():
        x2, y2, w2 = ref_paths[srd_path_name][0], ref_paths[srd_path_name][1], ref_paths[srd_path_name][4]
        x2_rot, y2_rot = rotate_aug(x2, y2, start_point, theta)
        plt.plot(x2_rot, y2_rot, linewidth=w2 * 36 * d // r, color='b')
        intersection_x_rot, intersection_y_rot = rotate_aug(intersection[0], intersection[1], start_point, theta)
        circle = patches.Circle((intersection_x_rot, intersection_y_rot), (w1 + w2) * 2 * d / r, color='g', zorder=30)
        axes.add_patch(circle)
    # set x y range
    plt.xlim(start_point[0] - r//2, start_point[0] + r//2)
    plt.ylim(start_point[1], start_point[1] + r)

    # remove the white biankuang
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.savefig(save_dir+'{}.png'.format(fig_name))
    plt.close()


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


def main(base_path, dir_name, work_dir):
    ref_paths, csv_dict, rare_paths = get_ref_paths(base_path, dir_name)
    rare_paths += ['6-4']
    path_names = sorted(ref_paths.keys())
    intersection_info = dict()
    for path_name in path_names:
        intersection_info[path_name] = dict()
    for i in range(len(path_names)):
        path1 = path_names[i]
        if path1 in rare_paths:
            continue
        for j in range(i + 1, len(path_names)):
            path2 = path_names[j]
            if path2 in rare_paths:
                continue
            seq1 = ref_paths[path1]
            seq2 = ref_paths[path2]
            if path1.split('-')[0] == path2.split('-')[0]:
                continue
            intersection = find_intersection(seq1, seq2)
            if intersection is None:
                continue
            else:
                # intersection of path1 and path2 exists
                intersection, first_id, second_id = intersection
                intersection_info[path1][path2] = intersection
                intersection_info[path2][path1] = intersection
                '''
                plot_intersection(seq1[0], seq1[1], seq1[4], seq2[0], seq2[1], seq2[4], path1+' '+path2,
                                  intersection, first_id, second_id)
                degree = 2
                # rotate x1,y1 +- degree around intersection and plot
                theta = math.pi*degree/180
                x1_rot, y1_rot = rotate_aug(seq1[0], seq1[1], intersection, theta)
                plot_intersection(x1_rot, y1_rot, seq1[4], seq2[0], seq2[1], seq2[4], path1+' '+path2+'_'+str(degree),
                                  intersection, first_id, second_id)

                x1_rot, y1_rot = rotate_aug(seq1[0], seq1[1], intersection, -theta)
                plot_intersection(x1_rot, y1_rot, seq1[4], seq2[0], seq2[1], seq2[4],
                                  path1 + ' ' + path2 + '_-' + str(degree),
                                  intersection, first_id, second_id)
                break
                '''

    ref_path_id_dict = dict()
    for i, path_name in enumerate(path_names):
        ref_path_id_dict[path_name] = i
    for csv_id, tracks in csv_dict.items():
        # for each csv, save a dict to pickle
        coordinate_dict = dict()
        print(csv_id)
        # plot global image
        cur_csv_work_dir = work_dir+'target_surrounding_images/'+csv_id+'/'
        # plot_global_image(ref_paths, '{}_global'.format(csv_id), work_dir, bg=True)
        for agent, path_name in tracks:
            if path_name in rare_paths:
                continue
            # for each agent(target car), save the agent to [ref_path_id][agent]
            ref_path_id = ref_path_id_dict[path_name]
            if ref_path_id not in coordinate_dict.keys():
                coordinate_dict[ref_path_id] = dict()
            coordinate_dict[ref_path_id][agent.track_id] = dict()
            start_id = int(path_name.split('-')[0])
            # select the starting frame from start to end, if in starting area, crop and save ref path image
            for ts in range(agent.time_stamp_ms_first, agent.time_stamp_ms_last, 100):
                coordinate_dict[ref_path_id][agent.track_id][ts] = dict()
                ms = agent.motion_states[ts]
                # judge if in starting area
                polygon_points = starting_area_dict[start_id]
                in_starting_area = judge_in_box(polygon_points['x'], polygon_points['y'], (ms.x, ms.y))
                if in_starting_area == 0:
                    continue
                theta = math.pi/2 - ms.psi_rad
                # rotate the ref path around current point, crop (-20,20), (0,40), and save
                # plot_rotate_crop(ref_paths, path_name, intersection_info[path_name],
                #                  '{}_{}_{}_{}'.format(csv_id, str(agent.track_id), path_name, str(ts)),
                #                  theta, [ms.x, ms.y], cur_csv_work_dir)
                # calculate the relative coordinates of other cars and judge
                for car, car_path_name in tracks:
                    if ts not in car.motion_states.keys():
                        continue
                    # get the motion state of other car and judge if they are in the bounding box
                    car_ms = car.motion_states[ts]
                    car_x_rot, car_y_rot = rotate_aug(car_ms.x, car_ms.y, [ms.x, ms.y], theta)
                    new_coor = (car_x_rot - ms.x, car_y_rot - ms.y)
                    if -20 < new_coor[0] < 20 and 0 < new_coor[1] < 40:
                        coordinate_dict[ref_path_id][agent.track_id][ts][car.track_id] = new_coor
        pickle_save_dir = work_dir+'pickle/'
        pickle_file = open(pickle_save_dir+'{}_relative_coordinate.pkl'.format(csv_id), 'wb')
        pickle.dump(coordinate_dict, pickle_file)
        pickle_file.close()
    return


def find_starting_area(work_dir):
    d = 8  # fig size
    r = 100  # range of x and y
    fig, axes = plt.subplots(1, 1, figsize=(d, d), dpi=200)
    lanelet_map_file = "D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/DR_USA_Intersection_MA.osm"
    map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, 0, 0)
    x = [1006, 996.2, 1011.4, 1011.1]
    y = [1019, 1010, 1009.9, 1019.2]

    plt.plot(x[0:2], y[0:2], c='r', zorder=40)
    plt.plot(x[1:3], y[1:3], c='r', zorder=40)
    plt.plot(x[2:4], y[2:4], c='r', zorder=40)
    plt.plot(x[3:]+x[0:1], y[3:]+y[0:1], c='r', zorder=40)
    xs, ys = 980, 950
    plt.xlim(xs, xs + r)
    plt.ylim(ys, ys + r)
    # remove the white biankuang
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    save_dir = work_dir + 'global_images/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.savefig(save_dir+'starting_area.png')
    plt.close()


def plot_starting_area():
    d = 8  # fig size
    r = 100  # range of x and y
    fig, axes = plt.subplots(1, 1, figsize=(d, d), dpi=200)
    lanelet_map_file = "D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/DR_USA_Intersection_MA.osm"
    map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, 0, 0)
    for key, v in starting_area_dict.items():
        if 'stoplinex' in v:
            plt.plot(v['stoplinex'], v['stopliney'], zorder=40, c='g')
            k = (v['stopliney'][1]-v['stopliney'][0])/(v['stoplinex'][1]-v['stoplinex'][0])
            b = (v['stoplinex'][1]*v['stopliney'][0]-v['stoplinex'][0]*v['stopliney'][1])/(v['stoplinex'][1]-v['stoplinex'][0])
        x = v['x']
        y = v['y']
        plt.plot(x[0:2], y[0:2], c='r', zorder=40)
        plt.plot(x[1:3], y[1:3], c='r', zorder=40)
        plt.plot(x[2:4], y[2:4], c='r', zorder=40)
        plt.plot(x[3:] + x[0:1], y[3:] + y[0:1], c='r', zorder=40)
    xs, ys = 980, 950
    plt.xlim(xs, xs + r)
    plt.ylim(ys, ys + r)
    # remove the white biankuang
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    save_dir = work_dir + 'global_images/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.savefig(save_dir+'all_starting_area.png')
    plt.close()


if __name__ == '__main__':
    base_path = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/'
    dir_name = 'DR_USA_Intersection_MA/'
    work_dir = 'D:/Dev/UCB task/'
    main(base_path, dir_name, work_dir)
    # plot_starting_area()
