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


def plot_start_end_area(ax, starting_area_dict, end_area_dict):
    for key, v in starting_area_dict.items():
        x = v['x']
        y = v['y']
        stopline = v['stopline']
        stop_x = [st[0] for st in stopline]
        stop_y = [st[1] for st in stopline]
        if key == 1:
            k = (stop_y[1] - stop_y[0]) / (stop_x[1] - stop_x[0])
            b = (stop_x[1] * stop_y[0] - stop_x[0] * stop_y[1]) / (stop_x[1] - stop_x[0])
            b1 = b + 6 * (1 + k ** 2) ** 0.5
            b2 = b - 2 * (1 + k ** 2) ** 0.5
            b3 = b - 6 * (1 + k ** 2) ** 0.5
            b4 = b + 2 * (1 + k ** 2) ** 0.5
            x1 = np.array([i for i in range(930, 1070)])
            ax.plot(x1, k * x1 + b1, linewidth=5)
            ax.plot(x1, k * x1 + b2, linewidth=5)
            ax.plot(x1, k * x1 + b3, linewidth=5)
            ax.plot(x1, k * x1 + b4, linewidth=5)
        ax.plot(stop_x, stop_y, c='g', linewidth=5, zorder=30)
        ax.text(x[0], y[0], key, fontsize=20)
        ax.plot(x[0:2], y[0:2], c='r', zorder=40)
        ax.plot(x[1:3], y[1:3], c='r', zorder=40)
        ax.plot(x[2:4], y[2:4], c='r', zorder=40)
        ax.plot(x[3:] + x[0:1], y[3:] + y[0:1], c='r', zorder=40)
    for key, v in end_area_dict.items():
        x = v['x']
        y = v['y']
        ax.text(x[0], y[0], key, fontsize=20)
        ax.plot(x[0:2], y[0:2], c='r', zorder=40)
        ax.plot(x[1:3], y[1:3], c='r', zorder=40)
        ax.plot(x[2:4], y[2:4], c='r', zorder=40)
        ax.plot(x[3:] + x[0:1], y[3:] + y[0:1], c='r', zorder=40)


def plot_raw_ref_path(map_file, all_points, circle_point):
    fig, axes = plt.subplots(1, 1, figsize=(30, 20), dpi=100)
    map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
    for way_points in all_points[0, :]:
        x = [p[0] for p in way_points]
        y = [p[1] for p in way_points]
        plt.plot(x, y, linewidth=4)

    for p in circle_point:
        if math.isnan(p[0][0]):
            continue
        circle = patches.Circle(p[0], 1, color='r', zorder=3)
        axes.add_patch(circle)
    fig.canvas.mpl_connect('button_press_event', on_press)
    plt.show()


def plot_ref_path(map_file, ref_path_points, starting_area_dict, end_area_dict):
    fig, axes = plt.subplots(1, 1)
    map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
    keys = sorted(ref_path_points.keys())
    for k in keys:
        v = ref_path_points[k]
        xp = [p[0] for p in v]
        yp = [p[1] for p in v]
        plt.plot(xp, yp, linewidth=4)
    plot_start_end_area(axes, starting_area_dict, end_area_dict)
    fig.canvas.mpl_connect('button_press_event', on_press)
    plt.show()


def find_all_split_points(ref_paths):
    split_points = dict()
    path_names = sorted(ref_paths.keys())
    for path_name in path_names:
        split_points[path_name] = dict()
    for path1 in path_names:
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


def save_split_bg_figs(ref_paths, split, map_file, save_dir):
    keys = sorted(ref_paths.keys())
    fig_n = 0
    for i, path1 in enumerate(keys):
        if len(split[path1].keys()) == 0:
            continue
        for j in range(i + 1, len(keys)):
            path2 = keys[j]
            if path2 not in split[path1].keys():
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
            p, _, _, cnt = split[path1][path2]
            circle = patches.Circle(p, 0.8, color=(1, 0, 0),
                                    zorder=3, label=cnt)
            axes.add_patch(circle)
            plt.legend(prop={'size': 20})
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                print('make dir: ', save_dir)
            plt.savefig(save_dir + '{}.png'.format(path1 + '_' + path2))
            plt.close()
            fig_n += 1
            print(fig_n)


def save_intersection_bg_figs(ref_paths, intersections, map_file, save_dir):
    keys = sorted(ref_paths.keys())
    fig_n = 0
    for i, path1 in enumerate(keys):
        if len(intersections[path1].keys()) == 0:
            continue
        for j in range(i+1, len(keys)):
            path2 = keys[j]
            if path2 not in intersections[path1].keys():
                continue
            m_c = intersections[path1][path2]
            for k, (p, _, _, cnt) in enumerate(m_c):
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
                circle = patches.Circle(p, 0.8, color=(1 - k * 0.2, 0, 0),
                                        zorder=3, label=cnt)
                axes.add_patch(circle)
                plt.legend(prop={'size': 20})
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                    print('make dir: ', save_dir)
                plt.savefig(save_dir+'{}.png'.format(path1+'_'+path2+'_'+str(k)))
                plt.close()
                fig_n += 1
                print(fig_n)


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


def rotate_crop_2path_fig(ref_paths, path1, path2, theta, intersection_info, save_dir,
                          k, ref_frenet, n, col_id=(0, 1)):
    its, first_id, second_id, label = intersection_info
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
    if abs(xp2[-1]-its[0]) < 10 and abs(yp2[-1]-its[1]) < 10:
        # merging and the other is longer
        end1_out = abs(xp1[-1]-its[0]) > 10 or abs(yp1[-1]-its[1]) > 10
        if 'merging' in label and end1_out:
            append_x, append_y = get_append(xp2[-1], yp2[-1], xp1, yp1, 'end')
            xp2 = xp2 + append_x
            yp2 = yp2 + append_y
        # complete along the tangle
        else:
            xp2, yp2 = ref_path_completion(xp2, yp2, its[1] + r // 2, its[1] - r // 2,
                                           its[0] - r // 2, its[0] + r // 2, 'end')
    if abs(xp1[-1]-its[0]) < 10 and abs(yp1[-1]-its[1]) < 10:
        # merging and the other is longer
        end2_out = abs(xp2[-1] - its[0]) > 10 or abs(yp2[-1] - its[1]) > 10
        if 'merging' in label and end2_out:
            append_x, append_y = get_append(xp1[-1], yp1[-1], xp2, yp2, 'end')
            xp1 = xp1 + append_x
            yp1 = yp1 + append_y
        # complete along the tangle
        else:
            xp1, yp1 = ref_path_completion(xp1, yp1, its[1] + r // 2, its[1] - r // 2,
                                           its[0] - r // 2, its[0] + r // 2, 'end')
    # complete start part
    if abs(xp2[0]-its[0]) < 10 and abs(yp2[0]-its[1]) < 10:
        # split and the other is longer
        start1_out = abs(xp1[0] - its[0]) > 10 or abs(yp1[0] - its[1]) > 10
        if 'split' in label and start1_out:
            append_x, append_y = get_append(xp2[0], yp2[0], xp1, yp1, 'start')
            xp2 = append_x + xp2
            yp2 = append_y + yp2
        # complete along the tangle
        else:
            xp2, yp2 = ref_path_completion(xp2, yp2, its[1] + r // 2, its[1] - r // 2,
                                           its[0] - r // 2, its[0] + r // 2, 'start')
    if abs(xp1[0]-its[0]) < 10 and abs(yp1[0]-its[1]) < 10:
        # split and the other is longer
        start2_out = abs(xp2[0] - its[0]) > 10 or abs(yp2[0] - its[1]) > 10
        if 'split' in label and start2_out:
            append_x, append_y = get_append(xp1[0], yp1[0], xp2, yp2, 'start')
            xp1 = append_x + xp1
            yp1 = append_y + yp1
        # complete along the tangle
        else:
            xp1, yp1 = ref_path_completion(xp1, yp1, its[1] + r // 2, its[1] - r // 2,
                                           its[0] - r // 2, its[0] + r // 2, 'start')
    # rotate randomly to augment data
    xp1, yp1 = counterclockwise_rotate(xp1, yp1, its, theta)
    xp2, yp2 = counterclockwise_rotate(xp2, yp2, its, theta)
    if 'cross' in label:
        plt.plot(xp1[:first_id - start1], yp1[:first_id - start1], linewidth=lw, color=(col_id[0], 0, 1))
        plt.plot(xp2[:second_id - start2], yp2[:second_id - start2], linewidth=lw, color=(col_id[1], 0, 1), zorder=5)
        plt.plot(xp1[first_id - start1:], yp1[first_id - start1:], linewidth=lw, color=(col_id[0], 1, 1))
        plt.plot(xp2[second_id - start2:], yp2[second_id - start2:], linewidth=lw, color=(col_id[1], 1, 1), zorder=5)
    elif 'merging' in label:
        plt.plot(xp1[:first_id - start1], yp1[:first_id - start1], linewidth=lw, color=(col_id[0], 0, 1))
        plt.plot(xp2[:second_id - start2], yp2[:second_id - start2], linewidth=lw, color=(col_id[1], 0, 1), zorder=5)
        plt.plot(xp1[first_id - start1:], yp1[first_id - start1:], linewidth=lw, color=(0.5, 1, 1))
        plt.plot(xp2[second_id - start2:], yp2[second_id - start2:], linewidth=lw, color=(0.5, 1, 1), zorder=5)
    elif 'split' in label:
        plt.plot(xp1[first_id - start1:], yp1[first_id - start1:], linewidth=lw, color=(col_id[0], 1, 1))
        plt.plot(xp2[second_id - start2:], yp2[second_id - start2:], linewidth=lw, color=(col_id[1], 1, 1), zorder=5)
        plt.plot(xp1[:first_id - start1], yp1[:first_id - start1], linewidth=lw, color=(0.5, 0, 1))
        plt.plot(xp2[:second_id - start2], yp2[:second_id - start2], linewidth=lw, color=(0.5, 0, 1), zorder=5)
    # set x y range
    plt.xlim(its[0] - r // 2, its[0] + r // 2)
    plt.ylim(its[1] - r // 2, its[1] + r // 2)
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
                                      save_dir, k, ref_frenet, 0, (0, 1))
                # swap the colors
                rotate_crop_2path_fig(ref_paths, path1, path2, 0.0, its_info,
                                      save_dir, k, ref_frenet, 1, (1, 0))
                for r_n in range(rotate_n):
                    theta = random.random() * 2 * math.pi
                    rotate_crop_2path_fig(ref_paths, path1, path2, theta, its_info,
                                          save_dir, k, ref_frenet, 2*(r_n+1), (0, 1))
                    # swap the colors
                    rotate_crop_2path_fig(ref_paths, path1, path2, theta, its_info,
                                          save_dir, k, ref_frenet, 2*(r_n+1)+1, (1, 0))
                count += (1+rotate_n)*2
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
                                  save_dir, 0, ref_frenet, 0, (0, 1))
            # swap the colors
            rotate_crop_2path_fig(ref_paths, path1, path2, 0.0, its_info,
                                  save_dir, 0, ref_frenet, 1, (1, 0))
            for r_n in range(rotate_n):
                theta = random.random() * 2 * math.pi
                rotate_crop_2path_fig(ref_paths, path1, path2, theta, its_info,
                                      save_dir, 0, ref_frenet, 2*(r_n + 1), (0, 1))
                # swap the colors
                rotate_crop_2path_fig(ref_paths, path1, path2, theta, its_info,
                                      save_dir, 0, ref_frenet, 2*(r_n + 1)+1, (1, 0))
            count += (1+rotate_n)*2
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


def judge_start(track, areas):
    # judge if in some starting area frame by frame
    for ts in range(track.time_stamp_ms_first, track.time_stamp_ms_last+100, 100):
        motion_state = track['motion_states'][ts]
        cur_p = (motion_state['x'], motion_state['y'])
        for k, v in areas.items():
            in_box = judge_in_box(v['x'], v['y'], cur_p)
            if in_box == 1:
                return k
    return 0


def judge_end(track, areas):
    # judge if in some starting area frame by frame
    for ts in range(track.time_stamp_ms_first, track.time_stamp_ms_last+100, 100):
        motion_state = track['motion_states'][ts]
        cur_p = (motion_state['x'], motion_state['y'])
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
        csv_agents = dict()
        for agent in tracks:
            start_area = judge_start(agent, starting_areas)
            end_area = judge_end(agent, end_areas)
            if start_area == 0 or end_area == 0:
                print(agent.track_id, 'starting or ending area is 0, drop')
                continue
            path_name = str(start_area) + '-' + str(end_area)
            if path_name not in ref_path_points:
                path_name = str(start_area) + '--1-' + str(end_area)
            xy_points = ref_path_points[path_name]
            # calculate frenet s,d and velocity along s direction
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
            agent_dict = dict()
            agent_dict['track_id'] = agent.track_id
            agent_dict['time_stamp_ms_first'] = agent.time_stamp_ms_first
            agent_dict['time_stamp_ms_last'] = agent.time_stamp_ms_last
            agent_dict['ref path'] = path_name
            agent_dict['motion_states'] = dict()
            for ts, ms in agent.motion_states.items():
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
            csv_agents[agent.track_id] = agent_dict
        csv_dict[csv_name[-7:-4]] = csv_agents
    return csv_dict


def get_70_coor(track_data, start_ts):
    coors = []
    for coor_ts in range(start_ts, start_ts+70*100, 100):
        if coor_ts in track_data['motion_states'].keys():
            coor_ms = track_data['motion_states'][coor_ts]
            coors.append((coor_ms['x'], coor_ms['y']))
        else:
            coors.append('NaN')
    return coors


def save_all_edges(csv_dict, is_info, ref_frenet, starting_areas, split_points):
    all_edges = dict()
    for i, c_data in csv_dict.items():
        # get edge info in one csv file
        edges = get_csv_edges(c_data, is_info, ref_frenet, starting_areas, split_points)
        all_edges[i] = edges
        print(i, ': ', len(edges), '/', len(c_data))
    return all_edges


def get_csv_edges(c_data, is_info, ref_frenet, starting_areas, split_points):
    edges = dict()
    for ego_id, ego_data in c_data.items():
        # if in starting area and have at least 69 frames behind,
        # save ref path image and trajectory data
        ego_path = ego_data['ref path']
        for start_ts in range(ego_data['time_stamp_ms_first'],
                              ego_data['time_stamp_ms_last'] - 68 * 100, 100):
            # ego_start_ms = ego_track['motion_states'][start_ts]
            # start_id = int(ego_path.split('-')[0])
            # sx = starting_areas[start_id]['x']
            # sy = starting_areas[start_id]['y']
            # # if in starting area, [-6,2] from stopline
            # in_starting_area = judge_in_box(sx, sy, (ego_start_ms.x, ego_start_ms.y))
            # if in_starting_area == 0:
            #     continue
            if ego_id not in edges.keys():
                edges[ego_id] = dict()
            edges[ego_id][start_ts] = dict()
            # find other cars in 20th frame
            ts_20 = start_ts + 19 * 100
            ego_20_ms = ego_data['motion_states'][ts_20]
            # save x,y coordinate of all involved agents in 70 frames
            edges[ego_id][start_ts]['xy'] = dict()
            # id of involved agents in 20th frame
            edges[ego_id][start_ts]['agents'] = [ego_id]
            # intersections involved in 20th frame
            edges[ego_id][start_ts]['task'] = set()
            edges[ego_id][start_ts]['its_id'] = dict()
            case_its_s = dict()
            # find surrounding cars in [-10,10] according to 20th frame
            for other_id, other_data in c_data.items():
                # not self, and containing this timestamp
                if other_id == ego_id or ts_20 not in other_data['motion_states'].keys():
                    continue
                other_20_x = other_data['motion_states'][ts_20]['x']
                other_20_y = other_data['motion_states'][ts_20]['y']
                # delta of x,y in (-10, 10)
                if abs(other_20_x - ego_20_ms['x']) > 10 or abs(other_20_y - ego_20_ms['y']) > 10:
                    continue
                edges[ego_id][start_ts]['agents'].append(other_id)

            # delete agents without interaction with any other agent in this case
            to_delete_agents = []
            for id1 in edges[ego_id][start_ts]['agents']:
                interaction_num = 0

                ms1_20 = c_data[id1]['motion_states'][ts_20]
                path1 = c_data[id1]['ref path']
                # find interaction car2
                for id2 in edges[ego_id][start_ts]['agents']:
                    if id2 == id1:
                        continue
                    # the first frame agent1 and 2 appear together
                    start_ts_1_2 = max(start_ts, c_data[id1]['time_stamp_ms_first'],
                                       c_data[id2]['time_stamp_ms_first'])
                    ms1_start = c_data[id1]['motion_states'][start_ts_1_2]
                    ms2_start = c_data[id2]['motion_states'][start_ts_1_2]
                    path2 = c_data[id2]['ref path']
                    # the same path/intersection/splitting point
                    # have intersection and not arrived
                    if path2 in is_info[path1].keys():
                        # judge if having passed the intersection
                        intersections = is_info[path1][path2]
                        closest_k = -1
                        closest_dis = 1e10
                        ms2_20 = c_data[id2]['motion_states'][ts_20]
                        # find the nearest intersection of agent1 and agent2
                        for its_k, its in enumerate(intersections):
                            p = its[0]
                            dis = (p[0]-ms1_20['x'])**2+(p[1]-ms1_20['y'])**2
                            dis += (p[0]-ms2_20['x'])**2+(p[1]-ms2_20['y'])**2
                            if dis < closest_dis:
                                closest_dis = dis
                                closest_k = its_k

                        closest_its = intersections[closest_k]
                        its_s1 = ref_frenet[path1][closest_its[1]]
                        its_s2 = ref_frenet[path2][closest_its[2]]

                        # having passed the intersection in the 1st frame, no interaction
                        if ms1_start['frenet_s'] > its_s1 or ms2_start['frenet_s'] > its_s2:
                            continue
                        interaction_num += 1
                        if id1 not in edges[ego_id][start_ts]['its_id'].keys():
                            edges[ego_id][start_ts]['its_id'][id1] = dict()
                        edges[ego_id][start_ts]['its_id'][id1][id2] = closest_k
                        if id2 not in edges[ego_id][start_ts]['its_id'].keys():
                            edges[ego_id][start_ts]['its_id'][id2] = dict()
                        edges[ego_id][start_ts]['its_id'][id2][id1] = closest_k
                        if id1 not in case_its_s.keys():
                            case_its_s[id1] = dict()
                        case_its_s[id1][id2] = (its_s1, its_s2)
                        if id2 not in case_its_s.keys():
                            case_its_s[id2] = dict()
                        case_its_s[id2][id1] = (its_s2, its_s1)
                        pair = sorted([path1, path2])
                        task = pair[0] + '_' + pair[1] + '_' + str(closest_k)
                        edges[ego_id][start_ts]['task'].add(task)
                    # in the same ref path and ego behind other
                    elif path2 == path1:
                        interaction_num += 1
                        if id1 not in edges[ego_id][start_ts]['its_id'].keys():
                            edges[ego_id][start_ts]['its_id'][id1] = dict()
                        edges[ego_id][start_ts]['its_id'][id1][id2] = 0
                        if id2 not in edges[ego_id][start_ts]['its_id'].keys():
                            edges[ego_id][start_ts]['its_id'][id2] = dict()
                        edges[ego_id][start_ts]['its_id'][id2][id1] = 0
                        task = path1 + '_' + path2 + '_0'
                        edges[ego_id][start_ts]['task'].add(task)
                    # having the same starting area and different end area
                    # and the other car is ahead of ego car and not passed
                    elif path1.split('-')[0] == path2.split('-')[0]:
                        sp_point = split_points[path1][path2]
                        split_id1 = sp_point[1]
                        split_id2 = sp_point[2]
                        split_s1 = ref_frenet[path1][split_id1]
                        split_s2 = ref_frenet[path2][split_id2]

                        # having passed the split point in the 1st frame, no interaction
                        if split_s1 < ms1_start['frenet_s'] or split_s2 < ms2_start['frenet_s']:
                            continue
                        interaction_num += 1
                        if id1 not in edges[ego_id][start_ts]['its_id'].keys():
                            edges[ego_id][start_ts]['its_id'][id1] = dict()
                        edges[ego_id][start_ts]['its_id'][id1][id2] = 0
                        if id2 not in edges[ego_id][start_ts]['its_id'].keys():
                            edges[ego_id][start_ts]['its_id'][id2] = dict()
                        edges[ego_id][start_ts]['its_id'][id2][id1] = 0
                        pair = sorted([path1, path2])
                        task = pair[0] + '_' + pair[1] + '_0'
                        edges[ego_id][start_ts]['task'].add(task)
                    else:
                        pass
                # no interaction with any other agent
                if interaction_num == 0:
                    to_delete_agents.append(id1)
            # delete agents without interaction
            for del_agent in to_delete_agents:
                edges[ego_id][start_ts]['agents'].remove(del_agent)
            # delete no surrounding car cases
            if len(edges[ego_id][start_ts]['agents']) < 2:
                del edges[ego_id][start_ts]
                continue
            for agent_id in edges[ego_id][start_ts]['agents']:
                edges[ego_id][start_ts]['xy'][agent_id] = get_70_coor(c_data[agent_id], start_ts)

            # save relative x/y, frenet s to intersection/splitting point or delta s
            for cur_ts in range(start_ts, start_ts + 70 * 100, 100):
                edges[ego_id][start_ts][cur_ts] = dict()
                for id1 in edges[ego_id][start_ts]['agents']:
                    # car1 not in scenario this ts
                    if cur_ts not in c_data[id1]['motion_states'].keys():
                        edges[ego_id][start_ts][cur_ts][id1] = None
                        continue
                    if id1 not in edges[ego_id][start_ts][cur_ts].keys():
                        edges[ego_id][start_ts][cur_ts][id1] = dict()
                    cur_ms1 = c_data[id1]['motion_states'][cur_ts]
                    path1 = c_data[id1]['ref path']
                    theta1 = math.pi/2 - cur_ms1['psi_rad']
                    for id2 in edges[ego_id][start_ts]['agents']:
                        if id1 == id2:
                            continue
                        # car2 not in scenario in this timestamp
                        if cur_ts not in c_data[id2]['motion_states'].keys():
                            edges[ego_id][start_ts][cur_ts][id1][id2] = None
                            continue
                        cur_ms2 = c_data[id2]['motion_states'][cur_ts]
                        path2 = c_data[id2]['ref path']
                        if path1 == path2:
                            rel_x, rel_y = cur_ms2['x']-cur_ms1['x'], cur_ms2['y']-cur_ms1['y']
                            rot_x, rot_y = counterclockwise_rotate(rel_x, rel_y, (0, 0), theta1)
                            delta_s = cur_ms2['frenet_s'] - cur_ms1['frenet_s']
                            edges[ego_id][start_ts][cur_ts][id1][id2] = [(rot_x, rot_y), delta_s]
                        elif id1 in case_its_s.keys() and id2 in case_its_s[id1].keys():
                            rel_x, rel_y = cur_ms2['x'] - cur_ms1['x'], cur_ms2['y'] - cur_ms1['y']
                            rot_x, rot_y = counterclockwise_rotate(rel_x, rel_y, (0, 0), theta1)
                            its_s1, its_s2 = case_its_s[id1][id2]
                            s_1 = its_s1 - cur_ms1['frenet_s']
                            s_2 = its_s2 - cur_ms2['frenet_s']
                            edges[ego_id][start_ts][cur_ts][id1][id2] = [(rot_x, rot_y), s_1, s_2]
                        elif path1.split('-')[0] == path2.split('-')[0]:
                            rel_x, rel_y = cur_ms2['x'] - cur_ms1['x'], cur_ms2['y'] - cur_ms1['y']
                            rot_x, rot_y = counterclockwise_rotate(rel_x, rel_y, (0, 0), theta1)
                            sp_point = split_points[path1][path2]
                            sp_id1 = sp_point[1]
                            sp_id2 = sp_point[2]
                            sp_s1 = ref_frenet[path1][sp_id1]
                            sp_s2 = ref_frenet[path2][sp_id2]
                            s_1 = sp_s1 - cur_ms1['frenet_s']
                            s_2 = sp_s2 - cur_ms2['frenet_s']
                            edges[ego_id][start_ts][cur_ts][id1][id2] = [(rot_x, rot_y), s_1, s_2]

        if ego_id in edges.keys() and len(edges[ego_id]) == 0:
            del edges[ego_id]
    return edges
