import math
import numpy as np
from scipy.optimize import leastsq
import os
import random
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import map_vis_without_lanelet
from align_ref_img import counterclockwise_rotate
from utils.intersection_utils import find_intersection, find_split_point
from utils import dataset_reader
from utils import dict_utils
from utils.new_coor_ref_path_utils import judge_in_box
from utils.coordinate_transform import get_frenet


def residuals(p, x, y):
    a, b, r = p
    return r**2 - (y - b) ** 2 - (x - a) ** 2


def fit_circle(circle_point):
    raw_circle_x = []
    raw_circle_y = []
    for p in circle_point:
        if math.isnan(p[0][0]):
            continue
        raw_circle_x.append(p[0][0])
        raw_circle_y.append(p[0][1])
    x = np.array(raw_circle_x)
    y = np.array(raw_circle_y)
    result = leastsq(residuals, np.array([1, 1, 1]), args=(x, y))
    a, b, r = result[0]
    r = max(-r, r)
    print("a=", a, "b=", b, "r=", r)
    fit_circle_x = [a-r]
    fit_circle_y = [b]
    for i in range(359):
        nx, ny = counterclockwise_rotate(a-r, b, (a, b), (i+1)*math.pi/180)
        fit_circle_x.append(nx)
        fit_circle_y.append(ny)
    return fit_circle_x, fit_circle_y


def nearest_c_point(p, x_list, y_list):
    min_dis = 1e8
    xn, yn = 0, 0
    min_i = 0
    for i in range(len(x_list)):
        x, y = x_list[i], y_list[i]
        if (x-p[0])**2 + (y-p[1])**2 < min_dis:
            min_dis = (x-p[0])**2 + (y-p[1])**2
            xn, yn = x, y
            min_i = i
    return min_i, xn, yn


def find_all_split_points(ref_paths):
    split_points = dict()
    path_names = sorted(ref_paths.keys())
    for path_name in path_names:
        split_points[path_name] = dict()
    for i, path1 in enumerate(path_names):
        start_area1 = path1.split('-')[0]
        for path2 in path_names:
            if path1 == path2:
                continue
            start_area2 = path2.split('-')[0]
            # having the same starting area
            if start_area1 != start_area2:
                continue
            split_point = find_split_point(ref_paths[path1], ref_paths[path2])
            if split_point is not None:
                split_points[path1][path2] = split_point
    return split_points


def find_all_intersections(ref_paths):
    intersections = dict()
    path_names = sorted(ref_paths.keys())
    for path_name in path_names:
        intersections[path_name] = dict()
    for i, path1 in enumerate(path_names):
        for j in range(len(path_names)):
            path2 = path_names[j]
            if path1.split('-')[0] == path2.split('-')[0]:
                continue
            path1_x = ref_paths[path1][:, 0]
            path1_y = ref_paths[path1][:, 1]
            path2_x = ref_paths[path2][:, 0]
            path2_y = ref_paths[path2][:, 1]
            intersection = find_intersection(path1_x, path1_y, path2_x, path2_y,
                                             dis_th=2, mg_th=0.3)
            if intersection is not None and len(intersection) > 0:
                # intersection of path1 and path2 exists
                intersections[path1][path2] = intersection
    return intersections


def save_intersection_bg_figs(ref_paths, intersections, map_file, save_dir):
    keys = sorted(ref_paths.keys())
    for i, path1 in enumerate(keys):
        if len(intersections[path1].keys()) == 0:
            continue
        for j in range(i+1, len(keys)):
            path2 = keys[j]
            if path2 not in intersections[path1].keys():
                continue
            fig, axes = plt.subplots(1, 1, figsize=(30, 20), dpi=100)
            map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
            v = ref_paths[path1]
            xp = [p[0] for p in v]
            yp = [p[1] for p in v]
            plt.text(xp[0], yp[0], 'start', fontsize=20)
            plt.text(xp[-1], yp[-1], 'end', fontsize=20)
            plt.plot(xp, yp, linewidth=4, label=path1)
            v = ref_paths[path2]
            xp = [p[0] for p in v]
            yp = [p[1] for p in v]
            plt.text(xp[0], yp[0], 'start', fontsize=20)
            plt.text(xp[-1], yp[-1], 'end', fontsize=20)
            plt.plot(xp, yp, linewidth=4, label=path2)
            m_c = intersections[path1][path2]
            for k, (p, _, _, cnt) in enumerate(m_c):
                circle = patches.Circle(p, 0.8, color=(1-k*0.2, 0, 0),
                                        zorder=3, label=cnt)
                axes.add_patch(circle)
            plt.legend(prop={'size': 20})
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                print('make dir: ', save_dir)
            plt.savefig(save_dir+'{}.png'.format(path1+'_'+path2))
            plt.close()


def ref_path_completion(xp, yp, up, bottom, left, right, mode):
    if mode == 'start':
        x2, x1 = xp[0:2]
        y2, y1 = yp[0:2]
    else:
        x1, x2 = xp[-2:]
        y1, y2 = yp[-2:]
    k = (y2 - y1) / (x2 - x1)
    b = (x2 * y1 - x1 * y2) / (x2 - x1)
    if x1 > x2:
        x = x2 - 0.2
        while x > left:
            y = k * x + b
            if y > up or y < bottom:
                break
            if mode == 'start':
                xp = [x] + xp
                yp = [y] + yp
            else:
                xp = xp + [x]
                yp = yp + [y]
            x -= 0.2
        y = k * x + b
        if mode == 'start':
            xp = [x] + xp
            yp = [y] + yp
        else:
            xp = xp + [x]
            yp = yp + [y]
    else:
        x = x2 + 0.2
        while x < right:
            y = k * x + b
            if y > up or y < bottom:
                break
            if mode == 'start':
                xp = [x] + xp
                yp = [y] + yp
            else:
                xp = xp + [x]
                yp = yp + [y]
            x += 0.2
        y = k * x + b
        if mode == 'start':
            xp = [x] + xp
            yp = [y] + yp
        else:
            xp = xp + [x]
            yp = yp + [y]
    return xp, yp


def get_append(xp1, yp1, xps2, yps2, mode):
    # find the closest point of xp1, yp1 in xps2, yps2
    # and return the pre or post sequence to append with original sequence
    min_dis = 1e8
    closest_id = 0
    for dis_i, (x2, y2) in enumerate(zip(xps2, yps2)):
        dis = (x2 - xp1) ** 2 + (y2 - yp1) ** 2
        if dis < min_dis:
            closest_id = dis_i
            min_dis = dis
    if mode == 'end':
        append_x = [x for x in xps2[closest_id:]]
        append_y = [y for y in yps2[closest_id:]]
    else:
        append_x = [x for x in xps2[:closest_id]]
        append_y = [y for y in yps2[:closest_id]]
    return append_x, append_y


def rotate_crop_2path_fig(ref_paths, path1, path2, theta, path_info, save_dir,
                          k, ref_frenet, n):
    p, first_id, second_id, label = path_info
    d = 4  # fig size
    r = 20  # range of x and y
    dpi = 50
    fig, axes = plt.subplots(1, 1, figsize=(d, d), dpi=dpi)
    # set bg to black
    axes.patch.set_facecolor("k")
    v1 = ref_paths[path1]
    frenet1 = ref_frenet[path1]
    si1 = frenet1[first_id]
    start1 = 0
    for i, s in enumerate(frenet1):
        if s < si1-20:
            start1 = i
    end1 = len(v1)
    for i in range(len(frenet1)-1, -1, -1):
        s = frenet1[i]
        if s > si1+20:
            end1 = i
    xp1 = [p[0] for p in v1[start1:end1]]
    yp1 = [p[1] for p in v1[start1:end1]]

    lw = 2*72*d/r
    v2 = ref_paths[path2]
    frenet2 = ref_frenet[path2]
    si2 = frenet2[second_id]
    start2 = 0
    for i, s in enumerate(frenet2):
        if s < si2 - 20:
            start2 = i
    end2 = len(v2)
    for i in range(len(frenet2)-1, -1, -1):
        s = frenet2[i]
        if s > si2 + 20:
            end2 = i
    xp2 = [p[0] for p in v2[start2:end2]]
    yp2 = [p[1] for p in v2[start2:end2]]

    # complete end part
    if abs(xp2[-1]-p[0]) < 10 and abs(yp2[-1]-p[1]) < 10:
        # merging and the other is longer
        end1_out = abs(xp1[-1]-p[0]) > 10 or abs(yp1[-1]-p[1]) > 10
        if 'merging' in label and end1_out:
            append_x, append_y = get_append(xp2[-1], yp2[-1], xp1, yp1, 'end')
            xp2 = xp2 + append_x
            yp2 = yp2 + append_y
        # complete along the tangle
        else:
            xp2, yp2 = ref_path_completion(xp2, yp2, p[1] + r // 2, p[1] - r // 2,
                                           p[0] - r // 2, p[0] + r // 2, 'end')
    if abs(xp1[-1]-p[0]) < 10 and abs(yp1[-1]-p[1]) < 10:
        # merging and the other is longer
        end2_out = abs(xp2[-1] - p[0]) > 10 or abs(yp2[-1] - p[1]) > 10
        if 'merging' in label and end2_out:
            append_x, append_y = get_append(xp1[-1], yp1[-1], xp2, yp2, 'end')
            xp1 = xp1 + append_x
            yp1 = yp1 + append_y
        # complete along the tangle
        else:
            xp1, yp1 = ref_path_completion(xp1, yp1, p[1] + r // 2, p[1] - r // 2,
                                           p[0] - r // 2, p[0] + r // 2, 'end')
    # complete start part
    if abs(xp2[0]-p[0]) < 10 and abs(yp2[0]-p[1]) < 10:
        # split and the other is longer
        start1_out = abs(xp1[0] - p[0]) > 10 or abs(yp1[0] - p[1]) > 10
        if 'split' in label and start1_out:
            append_x, append_y = get_append(xp2[0], yp2[0], xp1, yp1, 'start')
            xp2 = append_x + xp2
            yp2 = append_y + yp2
        # complete along the tangle
        else:
            xp2, yp2 = ref_path_completion(xp2, yp2, p[1] + r // 2, p[1] - r // 2,
                                           p[0] - r // 2, p[0] + r // 2, 'start')
    if abs(xp1[0]-p[0]) < 10 and abs(yp1[0]-p[1]) < 10:
        # split and the other is longer
        start2_out = abs(xp2[0] - p[0]) > 10 or abs(yp2[0] - p[1]) > 10
        if 'split' in label and start2_out:
            append_x, append_y = get_append(xp1[0], yp1[0], xp2, yp2, 'start')
            xp1 = append_x + xp1
            yp1 = append_y + yp1
        # complete along the tangle
        else:
            xp1, yp1 = ref_path_completion(xp1, yp1, p[1] + r // 2, p[1] - r // 2,
                                           p[0] - r // 2, p[0] + r // 2, 'start')
    # rotate
    xp1, yp1 = counterclockwise_rotate(xp1, yp1, p, theta)
    xp2, yp2 = counterclockwise_rotate(xp2, yp2, p, theta)
    if 'cross' in label:
        plt.plot(xp1[:first_id - start1], yp1[:first_id - start1], linewidth=lw, color=(0, 0, 1))
        plt.plot(xp2[:second_id - start2], yp2[:second_id - start2], linewidth=lw, color=(1, 0, 1), zorder=5)
        plt.plot(xp1[first_id - start1:], yp1[first_id - start1:], linewidth=lw, color=(0, 1, 1))
        plt.plot(xp2[second_id - start2:], yp2[second_id - start2:], linewidth=lw, color=(1, 1, 1), zorder=5)
    elif 'merging' in label:
        plt.plot(xp1[:first_id - start1], yp1[:first_id - start1], linewidth=lw, color=(0, 0, 1))
        plt.plot(xp2[:second_id - start2], yp2[:second_id - start2], linewidth=lw, color=(1, 0, 1), zorder=5)
        plt.plot(xp1[first_id - start1:], yp1[first_id - start1:], linewidth=lw, color=(0.5, 1, 1))
        plt.plot(xp2[second_id - start2:], yp2[second_id - start2:], linewidth=lw, color=(0.5, 1, 1), zorder=5)
    elif 'split' in label:
        plt.plot(xp1[first_id - start1:], yp1[first_id - start1:], linewidth=lw, color=(0, 1, 1))
        plt.plot(xp2[second_id - start2:], yp2[second_id - start2:], linewidth=lw, color=(1, 1, 1), zorder=5)
        plt.plot(xp1[:first_id - start1], yp1[:first_id - start1], linewidth=lw, color=(0.5, 0, 1))
        plt.plot(xp2[:second_id - start2], yp2[:second_id - start2], linewidth=lw, color=(0.5, 0, 1), zorder=5)
    # set x y range
    plt.xlim(p[0] - r // 2, p[0] + r // 2)
    plt.ylim(p[1] - r // 2, p[1] + r // 2)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print('make dir: ', save_dir)
    # remove the white frame
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_dir + '{}_{}_{}_{}.png'.format(path1, path2, k, n))
    plt.close()


def crop_intersection_figs(ref_paths, intersections, ref_frenet, save_dir, rotate_n):
    keys = sorted(ref_paths.keys())
    random.seed(123)
    count = 0
    for i, path1 in enumerate(keys):
        if len(intersections[path1].keys()) == 0:
            continue
        for j in range(i + 1, len(keys)):
            path2 = keys[j]
            if path2 not in intersections[path1].keys():
                continue

            m_c = intersections[path1][path2]
            for k, its_info in enumerate(m_c):
                print(path1, path2)
                rotate_crop_2path_fig(ref_paths, path1, path2, 0.0, its_info,
                                      save_dir, k, ref_frenet, 0)
                for r_n in range(rotate_n):
                    theta = random.random() * 2 * math.pi
                    rotate_crop_2path_fig(ref_paths, path1, path2, theta, its_info,
                                          save_dir, k, ref_frenet, r_n+1)
                count += 1+rotate_n
                print(count)
    return


def crop_split_figs(ref_paths, split_points, ref_frenet, save_dir, rotate_n):
    keys = sorted(ref_paths.keys())
    random.seed(123)
    count = 0
    for i, path1 in enumerate(keys):
        if len(split_points[path1].keys()) == 0:
            continue
        for j in range(i + 1, len(keys)):
            path2 = keys[j]
            if path2 not in split_points[path1].keys():
                continue
            its_info = split_points[path1][path2]
            rotate_crop_2path_fig(ref_paths, path1, path2, 0.0, its_info,
                                  save_dir, 0, ref_frenet, 0)

            for r_n in range(rotate_n):
                theta = random.random() * 2 * math.pi
                rotate_crop_2path_fig(ref_paths, path1, path2, theta, its_info,
                                      save_dir, 0, ref_frenet, r_n + 1)
            count += 1 + rotate_n
            print(count)
    return


def get_ref_path(data, cx, cy):
    ref_path_points = dict()
    para_path = data['para_path'][0]
    circle_merge_point = data['circle_merge_point'][0]
    branch_id = data['branchID'][0]
    pre = dict()
    post = dict()

    for i in range(len(branch_id)):
        s = branch_id[i][0]
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


def on_press(event):
    print("my position:", event.button, event.xdata, event.ydata)


def judge_start(track, areas):
    # judge if in some starting area frame by frame
    for ts in range(track.time_stamp_ms_first, track.time_stamp_ms_last+100, 100):
        motion_state = track.motion_states[ts]
        cur_p = (motion_state.x, motion_state.y)
        for k, v in areas.items():
            in_box = judge_in_box(v['x'], v['y'], cur_p)
            if in_box == 1:
                return k
    return 0


def judge_end(track, areas):
    # judge if in some starting area frame by frame
    for ts in range(track.time_stamp_ms_first, track.time_stamp_ms_last+100, 100):
        motion_state = track.motion_states[ts]
        cur_p = (motion_state.x, motion_state.y)
        for k, v in areas.items():
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


def get_track_label(dir_name, ref_path_points, ref_frenet, starting_areas, end_areas):
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
            start_area = judge_start(agent, starting_areas)
            end_area = judge_end(agent, end_areas)
            if start_area == 0 or end_area == 0:
                print(agent.track_id, 0)
                continue
            path_name = str(start_area) + '-' + str(end_area)
            if path_name not in ref_path_points:
                path_name = str(start_area) + '--1-' + str(end_area)
            xy_points = ref_path_points[path_name]
            for ts in range(agent.time_stamp_ms_first, agent.time_stamp_ms_last + 100, 100):
                x = agent.motion_states[ts].x
                y = agent.motion_states[ts].y
                psi_rad = agent.motion_states[ts].psi_rad
                f_s, f_d = get_frenet(x, y, psi_rad, xy_points, ref_frenet[path_name])
                agent.motion_states[ts].frenet_s = f_s
                agent.motion_states[ts].frenet_d = f_d
                if ts > agent.time_stamp_ms_first:
                    vs = (f_s - agent.motion_states[ts-100].frenet_s) / 0.1
                    agent.motion_states[ts].vs = vs
            agent.motion_states[agent.time_stamp_ms_first].vs = agent.motion_states[agent.time_stamp_ms_first+100].vs
            agent_path[agent.track_id] = {'data': agent, 'ref path': path_name}
        csv_dict[csv_name[-7:-4]] = agent_path
    return csv_dict


def get_70_coor(track_data, start_ts):
    coors = []
    for coor_ts in range(start_ts, start_ts+70*100, 100):
        if coor_ts in track_data.motion_states.keys():
            coor_ms = track_data.motion_states[coor_ts]
            coors.append((coor_ms.x, coor_ms.y))
        else:
            coors.append('NaN')
    return coors


def save_edges(csv_dict, is_info, ref_frenet, starting_areas, split_points):
    all_edges = dict()
    for i, c_data in csv_dict.items():
        print(i)
        edges = dict()
        for ego_id, ego_data in c_data.items():
            ego_track = ego_data['data']
            ego_path = ego_data['ref path']
            # if in starting area and have at least 69 frames behind,
            # save ref path image and trajectory data
            for start_ts in range(ego_track.time_stamp_ms_first,
                                  ego_track.time_stamp_ms_last - 68 * 100, 100):
                ego_start_ms = ego_track.motion_states[start_ts]
                start_id = int(ego_path.split('-')[0])
                sx = starting_areas[start_id]['x']
                sy = starting_areas[start_id]['y']
                in_starting_area = judge_in_box(sx, sy, (ego_start_ms.x, ego_start_ms.y))
                if in_starting_area == 0:
                    continue
                if ego_id not in edges.keys():
                    edges[ego_id] = dict()
                edges[ego_id][start_ts] = dict()
                ts_20 = start_ts + 19 * 100
                ego_20_ms = ego_track.motion_states[ts_20]

                # save x,y coordinate of all involved agents in 70 frames
                edges[ego_id][start_ts]['coordinate'] = dict()
                # id of involved agents in 20th frame
                edges[ego_id][start_ts]['agents'] = [ego_id]
                # intersections involved in 20th frame
                edges[ego_id][start_ts]['task'] = []
                edges[ego_id][start_ts]['coordinate'][ego_id] = get_70_coor(ego_track, start_ts)

                for other_id, other_data in c_data.items():
                    other_track = other_data['data']
                    other_path = other_data['ref path']
                    # not self, and containing this timestamp
                    if other_id == ego_id or ts_20 not in other_track.motion_states.keys():
                        continue
                    other_20_x = other_track.motion_states[ts_20].x
                    other_20_y = other_track.motion_states[ts_20].y
                    # delta of x,y in (-10, 10)
                    if abs(other_20_x - ego_20_ms.x) > 10 or abs(other_20_y - ego_20_ms.y) > 10:
                        continue
                    # have intersection
                    if other_path in is_info[ego_path].keys():
                        edges[ego_id][start_ts]['agents'].append(other_id)
                        edges[ego_id][start_ts]['coordinate'][other_id] = get_70_coor(other_track, start_ts)
                        pair = sorted([ego_path, other_path])
                        edges[ego_id][start_ts]['task'].append((pair[0], pair[1]))
                    # in the same ref path and ego behind other
                    elif other_path == ego_path and ego_track.motion_states[ts_20].frenet_s < \
                            other_track.motion_states[ts_20].frenet_s:
                        edges[ego_id][start_ts]['agents'].append(other_id)
                        edges[ego_id][start_ts]['coordinate'][other_id] = get_70_coor(other_track, start_ts)
                    # having the same starting area
                    elif other_path.split('-')[0] == ego_path.split('-')[0]:
                        sp_point = split_points[ego_path][other_path]
                        if sp_point is not None:
                            sp_id1 = sp_point[1]
                            sp_id2 = sp_point[2]
                            sp_s1 = ref_frenet[ego_path][sp_id1]
                            sp_s2 = ref_frenet[other_path][sp_id2]
                            # having not arrived the split point
                            if sp_s1 > ego_track.motion_states[ts_20].frenet_s\
                                    or sp_s2 > other_track.motion_states[ts_20].frenet_s:
                                pass
                # delete no surrounding car cases
                if len(edges[ego_id][start_ts]['agents']) < 2:
                    del edges[ego_id][start_ts]
                    continue
                for cur_ts in range(start_ts, start_ts + 70 * 100, 100):
                    ego_cur_ms = ego_track.motion_states[cur_ts]
                    edges[ego_id][start_ts][cur_ts] = dict()
                    theta = math.pi/2 - ego_cur_ms.psi_rad
                    for other_id in edges[ego_id][start_ts]['agents'][1:]:
                        if cur_ts not in c_data[other_id]['data'].motion_states.keys():
                            continue
                        other_cur_ms = c_data[other_id]['data'].motion_states[cur_ts]
                        other_path = c_data[other_id]['ref path']
                        rot_x, rot_y = counterclockwise_rotate(other_cur_ms.x, other_cur_ms.y,
                                                               (ego_cur_ms.x, ego_cur_ms.y), theta)
                        edge_info = [(rot_x-ego_cur_ms.x, rot_y-ego_cur_ms.y)]
                        # the same path
                        if other_path == ego_path:
                            delta_s = other_cur_ms.frenet_s - ego_cur_ms.frenet_s
                            edge_info.append(delta_s)
                        # intersection
                        else:
                            intersections = is_info[ego_path][other_path]
                            closest_its = None
                            closest_dis = 1e8
                            # find the closest intersection
                            for its in intersections:
                                p, _, _, _ = its
                                dis = (p[0] - ego_cur_ms.x) ** 2 + (p[1] - ego_cur_ms.y) ** 2
                                if dis < closest_dis:
                                    closest_dis = dis
                                    closest_its = its
                            _, first_id, second_id, _ = closest_its
                            its_s1 = ref_frenet[ego_path][first_id]
                            its_s2 = ref_frenet[other_path][second_id]
                            s_ego = its_s1 - ego_cur_ms.frenet_s
                            s_other = its_s2 - other_cur_ms.frenet_s
                            edge_info += [s_ego, s_other]
                        edges[ego_id][start_ts][cur_ts][other_id] = edge_info
            if ego_id in edges.keys() and len(edges[ego_id]) == 0:
                del edges[ego_id]
        all_edges[i] = edges
    return all_edges
