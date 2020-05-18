from utils import dataset_reader
from utils import dict_utils
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
import glob
import os


def judge_start(ms, track):
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
    # rotate the points
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
    csv_list = list()
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
        csv_list.append(agent_path)
    ref_paths = dict()
    beta_dict = {'7-8': math.pi / 4, '12-8': -math.pi / 6, '2-10': math.pi / 4, '2-11': math.pi / 4,
                 '6-1': -math.pi / 6, '3-4': -math.pi / 6, '9-4': math.pi / 4, '9-5': math.pi / 4,
                 '9-10': -math.pi / 6, '13-5': -math.pi / 6, '14-4': -math.pi / 6, '6-11': -math.pi / 6,
                 '7-10': 0, '2-8': 0, '3-8': 0, '9-1': 0, '12-5': -math.pi / 6, '13-8': -math.pi / 6,
                 '14-1': math.pi / 4, '14-15': -math.pi / 6, '13-4': math.pi / 4, '14-5': math.pi / 4,
                 '12-10': math.pi / 12, '7-11': -math.pi / 6, '9-11': -math.pi / 6, '13-10': math.pi / 12,
                 '2-4': -math.pi / 6, '3-5': -math.pi / 6, '6-10': -math.pi / 6, '6-4': 0}
    poly_dict = {'7-8': 6, '12-8': 6, '2-10': 6, '2-11': 6, '6-1': 6, '3-4': 6, '9-4': 6, '9-5': 6,
                 '9-10': 4, '13-5': 1, '14-4': 1, '6-11': 3, '7-10': 1, '2-8': 2, '3-8': 2, '9-1': 4,
                 '12-5': 1, '13-8': 6, '14-1': 5, '14-15': 1, '13-4': 1, '14-5': 6, '12-10': 4, '7-11': 4,
                 '9-11': 4, '13-10': 6, '2-4': 4, '3-5': 4, '6-10': 4, '6-4': 4}
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
    return ref_paths, csv_list, rare_paths


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
    plt.plot(x1, y1, linewidth=w1 * 72 * d // r, color='b')
    plt.plot(x2, y2, linewidth=w2 * 72 * d // r, color='g')
    circle = patches.Circle(intersection, (w1 + w2) * 2 * d / r, color='r', zorder=3)
    axes.add_patch(circle)
    # draw the arrow whose length is al
    al = 8
    axes.arrow(x1[id1], y1[id1], al/(k1_rot**2+1)**0.5, al * k1_rot/(k1_rot**2+1)**0.5, zorder=4,
               color='purple', width=0.2, head_width=0.6)
    axes.arrow(x2[id2], y2[id2], al/(k2_rot**2+1)**0.5, al * k2_rot/(k2_rot**2+1)**0.5, zorder=5,
               color='yellow', width=0.2, head_width=0.6)
    axes.arrow(x1[id1], y1[id1], al / (k1 ** 2 + 1) ** 0.5, al * k1 / (k1 ** 2 + 1) ** 0.5, zorder=4,
               color='purple', width=0.2, head_width=0.6)
    axes.arrow(x2[id2], y2[id2], al / (k2 ** 2 + 1) ** 0.5, al * k2 / (k2 ** 2 + 1) ** 0.5, zorder=5,
               color='yellow', width=0.2, head_width=0.6)
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


def plot_global_image(ref_paths, path_name, path_info, fig_name):
    '''
    :param ref_paths:
    :param path_names: target path name in str
    :param path_info: a dict [path name]:intersection
    :param fig_name: figure name
    :return:
    '''
    d = 8  # fig size
    r = 100  # range of x and y
    fig, axes = plt.subplots(1, 1, figsize=(d, d), dpi=200)
    # set bg to black
    axes.patch.set_facecolor("k")
    x1, y1, w1 = ref_paths[path_name][0], ref_paths[path_name][1], ref_paths[path_name][4]
    plt.plot(x1, y1, linewidth=w1 * 72 * d // r, color='r')
    for srd_path_name, intersection in path_info.items():
        x2, y2, w2 = ref_paths[srd_path_name][0], ref_paths[srd_path_name][1], ref_paths[srd_path_name][4]
        plt.plot(x2, y2, linewidth=w2 * 72 * d // r, color='b')
        circle = patches.Circle(intersection, (w1 + w2) * 2 * d / r, color='g', zorder=3)
        axes.add_patch(circle)
    # set x y range
    xs, ys = 980, 950
    plt.xlim(xs, xs + r)
    plt.ylim(ys, ys + r)

    # remove the white biankuang
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig('D:/Dev/UCB task/global_images/{}.png'.format(fig_name))
    plt.close()


def plot_rotate_crop(ref_paths, path_name, path_info, fig_name, theta, start_point):
    d = 4  # fig size
    r = 40  # range of x and y
    dpi = 50
    fig, axes = plt.subplots(1, 1, figsize=(d, d), dpi=dpi)
    # set bg to black
    axes.patch.set_facecolor("k")
    x1, y1, w1 = ref_paths[path_name][0], ref_paths[path_name][1], ref_paths[path_name][4]
    x1_rot, y1_rot = rotate_aug(x1, y1, start_point, theta)
    plt.plot(x1_rot, y1_rot, linewidth=w1 * 72 * d // r, color='r')
    for srd_path_name, intersection in path_info.items():
        x2, y2, w2 = ref_paths[srd_path_name][0], ref_paths[srd_path_name][1], ref_paths[srd_path_name][4]
        x2_rot, y2_rot = rotate_aug(x2, y2, start_point, theta)
        plt.plot(x2_rot, y2_rot, linewidth=w2 * 72 * d // r, color='b')
        intersection_x_rot, intersection_y_rot = rotate_aug(intersection[0], intersection[1], start_point, theta)
        circle = patches.Circle((intersection_x_rot, intersection_y_rot), (w1 + w2) * 2 * d / r, color='g', zorder=3)
        axes.add_patch(circle)
    # set x y range
    plt.xlim(start_point[0] - r//2, start_point[0] + r//2)
    plt.ylim(start_point[1], start_point[1] + r)

    # remove the white biankuang
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig('D:/Dev/UCB task/target_surrounding_images/{}.png'.format(fig_name))
    plt.close()


def main(base_path, dir_name):
    ref_paths, csv_list, rare_paths = get_ref_paths(base_path, dir_name)
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

    for i, tracks in enumerate(csv_list):
        for agent, path_name in tracks:
            if path_name in rare_paths:
                continue
            # plot global image
            # plot_global_image(ref_paths, path_name, intersection_info[path_name], path_name+'_global')
            # from start to end, crop
            ms = agent.motion_states[agent.time_stamp_ms_first + 10 * 100]
            theta = math.pi/2 - ms.psi_rad
            plot_rotate_crop(ref_paths, path_name, intersection_info[path_name], str(i) + '_' + path_name+'_50',
                             theta, [ms.x, ms.y])
            # for ts in range(agent.time_stamp_ms_first, agent.time_stamp_ms_last, 100):
            #     # judge if meeting requirements of starting
            #     ms = agent.motion_states[ts]
        break
    return


if __name__ == '__main__':
    base_path = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/'
    dir_name = 'DR_USA_Intersection_MA/'
    main(base_path, dir_name)
