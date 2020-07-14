import math
import numpy as np
from scipy.optimize import leastsq
import os
import random
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import map_vis_without_lanelet
from align_ref_img import counterclockwise_rotate
from utils.intersection_utils import find_interaction, cal_dis
from utils import dataset_reader
from utils import dict_utils
from utils.coordinate_transform import get_frenet, closest_point_index


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
    x = y = None
    for key, v in starting_area_dict.items():
        x = v['x']
        y = v['y']
        stopline = v['stopline']
        stop_x = [st[0] for st in stopline]
        stop_y = [st[1] for st in stopline]
        if key == 11:
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
        ax.plot(stop_x, stop_y, c='g', linewidth=2, zorder=30)
        ax.text(x[0], y[0], key, fontsize=20, zorder=41)
        ax.plot(x[0:2], y[0:2], c='g', zorder=40)
        ax.plot(x[1:3], y[1:3], c='g', zorder=40)
        ax.plot(x[2:4], y[2:4], c='g', zorder=40)
        ax.plot(x[3:] + x[0:1], y[3:] + y[0:1], c='g', zorder=40)
    ax.plot(x[0:2], y[0:2], c='g', zorder=40, label='start')
    for key, v in end_area_dict.items():
        x = v['x']
        y = v['y']
        ax.text(x[0], y[0], key, fontsize=20, zorder=41)
        ax.plot(x[0:2], y[0:2], c='r', zorder=40)
        ax.plot(x[1:3], y[1:3], c='r', zorder=40)
        ax.plot(x[2:4], y[2:4], c='r', zorder=40)
        ax.plot(x[3:] + x[0:1], y[3:] + y[0:1], c='r', zorder=40)
    ax.plot(x[0:2], y[0:2], c='r', zorder=40, label='end')


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
        plt.plot(xp, yp, linewidth=2)
    plot_start_end_area(axes, starting_area_dict, end_area_dict)
    fig.canvas.mpl_connect('button_press_event', on_press)
    plt.legend()
    plt.show()


def find_all_interactions(ref_paths, th=0.6, skip=20):
    interactions = dict()
    path_names = sorted(ref_paths.keys())
    for path_name in path_names:
        interactions[path_name] = dict()
    for i, path1 in enumerate(path_names):
        for j in range(i+1, len(path_names)):
            path2 = path_names[j]
            interaction12, interaction21 = find_interaction(ref_paths[path1], ref_paths[path2], th, skip)
            if interaction12 is not None and len(interaction12) > 0:
                # interaction of path1 and path2 exists
                interactions[path1][path2] = interaction12
                interactions[path2][path1] = interaction21
    return interactions


def save_complete_ref_path_fig(ref_paths, save_dir, xlim, ylim):
    keys = sorted(ref_paths.keys())
    for path1 in keys:
        fig, axes = plt.subplots(1, 1, figsize=((xlim[1]-xlim[0])//5, (ylim[1]-ylim[0])//5), dpi=8)
        # set bg to black
        axes.patch.set_facecolor("k")
        lw = 10
        plt.plot([p[0] for p in ref_paths[path1]], [p[1] for p in ref_paths[path1]], linewidth=lw)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print('make dir: ', save_dir)
        # remove the white frame
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(save_dir + '{}.png'.format(path1))
        plt.close()


def save_interaction_bg_figs(ref_paths, interactions, map_file, save_dir):
    keys = sorted(ref_paths.keys())
    fig_n = 0
    for i, path1 in enumerate(keys):
        if len(interactions[path1].keys()) == 0:
            continue
        for j in range(i+1, len(keys)):
            path2 = keys[j]
            if path2 not in interactions[path1].keys():
                continue
            ita = interactions[path1][path2]
            for k, (p, _, _, label) in enumerate(ita):
                fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
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
                                        zorder=3, label=label)
                axes.add_patch(circle)
                plt.legend(prop={'size': 20})
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                    print('make dir: ', save_dir)
                plt.savefig(save_dir+'{}_{}_{}.png'.format(path1, path2, str(k)))
                plt.close()
                fig_n += 1
                print(path1, path2, fig_n)


def ref_path_completion(xp, yp, up, bottom, left, right, mode):
    if mode == 'start':
        x2, x1 = xp[0], xp[2]
        y2, y1 = yp[0], yp[2]
    else:
        x1, x2 = xp[-3], xp[-1]
        y1, y2 = yp[-3], yp[-1]
    k = (y2 - y1) / (x2 - x1)
    b = (x2 * y1 - x1 * y2) / (x2 - x1)
    if x1 > x2:
        x = x2 - 0.2
        while x > left-5:
            y = k * x + b
            if y > up+5 or y < bottom-5:
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
        while x < right+5:
            y = k * x + b
            if y > up+5 or y < bottom-5:
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


def rotate_crop_2path_fig(ref_paths, path1, path2, theta, interaction_info, save_dir,
                          k, ref_frenet, n, save_name=None, col_id=(0, 1)):
    its, first_id, second_id, label = interaction_info

    v1 = ref_paths[path1]
    v2 = ref_paths[path2]
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

    d = 6  # fig size
    r = 30  # range of x and y
    dpi = 8
    fig, axes = plt.subplots(1, 1, figsize=(d, d), dpi=dpi)
    # set bg to black
    axes.patch.set_facecolor("k")
    lw = 0.8*72*d/r
    # circle = patches.Circle(its, 2, color='r', zorder=10)
    # axes.add_patch(circle)
    # complete end part
    if abs(xp2[-1]-its[0]) < r/2 and abs(yp2[-1]-its[1]) < r/2:
        xp2, yp2 = ref_path_completion(xp2, yp2, its[1] + r / 2, its[1] - r / 2,
                                       its[0] - r / 2, its[0] + r / 2, 'end')
    if abs(xp1[-1]-its[0]) < r/2 and abs(yp1[-1]-its[1]) < r/2:
        xp1, yp1 = ref_path_completion(xp1, yp1, its[1] + r // 2, its[1] - r // 2,
                                       its[0] - r / 2, its[0] + r / 2, 'end')
    # complete start part
    if abs(xp2[0]-its[0]) < r/2 and abs(yp2[0]-its[1]) < r/2:
        xp2n, yp2n = ref_path_completion(xp2, yp2, its[1] + r // 2, its[1] - r // 2,
                                       its[0] - r / 2, its[0] + r / 2, 'start')
        second_id += len(xp2n) - len(xp2)
        xp2, yp2 = xp2n, yp2n
    if abs(xp1[0]-its[0]) < r/2 and abs(yp1[0]-its[1]) < r/2:
        xp1n, yp1n = ref_path_completion(xp1, yp1, its[1] + r // 2, its[1] - r // 2,
                                       its[0] - r / 2, its[0] + r / 2, 'start')
        first_id += len(xp1n) - len(xp1)
        xp1, yp1 = xp1n, yp1n

    # rotate randomly to augment data
    xp1, yp1 = counterclockwise_rotate(xp1, yp1, its, theta)
    xp2, yp2 = counterclockwise_rotate(xp2, yp2, its, theta)
    if 'cross' in label:
        if n % 2 == 0:
            plt.plot(xp1[:first_id - start1], yp1[:first_id - start1], linewidth=lw, color=(col_id[0], 0, 1))
            plt.plot(xp2[:second_id - start2], yp2[:second_id - start2], linewidth=lw, color=(col_id[1], 0, 1), zorder=5)
            plt.plot(xp1[first_id - start1:], yp1[first_id - start1:], linewidth=lw, color=(col_id[0], 1, 1))
            plt.plot(xp2[second_id - start2:], yp2[second_id - start2:], linewidth=lw, color=(col_id[1], 1, 1), zorder=5)
        else:
            plt.plot(xp1[:first_id - start1], yp1[:first_id - start1], linewidth=lw, color=(col_id[0], 0, 1),
                     zorder=5)
            plt.plot(xp2[:second_id - start2], yp2[:second_id - start2], linewidth=lw, color=(col_id[1], 0, 1))
            plt.plot(xp1[first_id - start1:], yp1[first_id - start1:], linewidth=lw, color=(col_id[0], 1, 1),
                     zorder=5)
            plt.plot(xp2[second_id - start2:], yp2[second_id - start2:], linewidth=lw, color=(col_id[1], 1, 1))
    elif 'merging' in label:
        if n % 2 == 0:
            plt.plot(xp1[:first_id - start1], yp1[:first_id - start1], linewidth=lw, color=(col_id[0], 0, 1))
            plt.plot(xp2[:second_id - start2], yp2[:second_id - start2], linewidth=lw, color=(col_id[1], 0, 1), zorder=5)
            plt.plot(xp1[first_id - start1:], yp1[first_id - start1:], linewidth=lw, color=(0.5, 1, 1))
            plt.plot(xp2[second_id - start2:], yp2[second_id - start2:], linewidth=lw, color=(0.5, 1, 1), zorder=5)
        else:
            plt.plot(xp1[:first_id - start1], yp1[:first_id - start1], linewidth=lw, color=(col_id[0], 0, 1),
                     zorder=5)
            plt.plot(xp2[:second_id - start2], yp2[:second_id - start2], linewidth=lw, color=(col_id[1], 0, 1))
            plt.plot(xp1[first_id - start1:], yp1[first_id - start1:], linewidth=lw, color=(0.5, 1, 1), zorder=5)
            plt.plot(xp2[second_id - start2:], yp2[second_id - start2:], linewidth=lw, color=(0.5, 1, 1))
    elif 'split' in label:
        if n % 2 == 0:
            plt.plot(xp1[first_id - start1:], yp1[first_id - start1:], linewidth=lw, color=(col_id[0], 1, 1))
            plt.plot(xp2[second_id - start2:], yp2[second_id - start2:], linewidth=lw, color=(col_id[1], 1, 1), zorder=5)
            plt.plot(xp1[:first_id - start1], yp1[:first_id - start1], linewidth=lw, color=(0.5, 0, 1))
            plt.plot(xp2[:second_id - start2], yp2[:second_id - start2], linewidth=lw, color=(0.5, 0, 1), zorder=5)
        else:
            plt.plot(xp1[first_id - start1:], yp1[first_id - start1:], linewidth=lw, color=(col_id[0], 1, 1),
                     zorder=5)
            plt.plot(xp2[second_id - start2:], yp2[second_id - start2:], linewidth=lw, color=(col_id[1], 1, 1))
            plt.plot(xp1[:first_id - start1], yp1[:first_id - start1], linewidth=lw, color=(0.5, 0, 1), zorder=5)
            plt.plot(xp2[:second_id - start2], yp2[:second_id - start2], linewidth=lw, color=(0.5, 0, 1))
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
    if save_name is None:
        save_name = '{}_{}_{}_{}.png'.format(path1, path2, k, n)
    plt.savefig(save_dir + save_name)
    plt.close()


def crop_interaction_figs(ref_paths, interactions, ref_frenet, save_dir, rotate_n):
    keys = sorted(ref_paths.keys())
    random.seed(123)
    count = 0
    for i, path1 in enumerate(keys):
        if len(interactions[path1].keys()) == 0:
            continue
        for j in range(i + 1, len(keys)):
            path2 = keys[j]
            if path2 not in interactions[path1].keys():
                continue
            ita = interactions[path1][path2]
            for k, ita_info in enumerate(ita):
                print(path1, path2, k)
                rotate_crop_2path_fig(ref_paths, path1, path2, 0.0, ita_info,
                                      save_dir, k, ref_frenet, 0, col_id=(0, 1))
                # swap the color order
                rotate_crop_2path_fig(ref_paths, path1, path2, 0.0, ita_info,
                                      save_dir, k, ref_frenet, 1, col_id=(0, 1))
                # swap the colors
                rotate_crop_2path_fig(ref_paths, path1, path2, 0.0, ita_info,
                                      save_dir, k, ref_frenet, 2, col_id=(1, 0))

                # swap the colors
                rotate_crop_2path_fig(ref_paths, path1, path2, 0.0, ita_info,
                                      save_dir, k, ref_frenet, 3, col_id=(1, 0))
                for r_n in range(rotate_n):
                    theta = random.random() * 2 * math.pi
                    rotate_crop_2path_fig(ref_paths, path1, path2, theta, ita_info,
                                          save_dir, k, ref_frenet, 4*(r_n+1), col_id=(0, 1))
                    rotate_crop_2path_fig(ref_paths, path1, path2, theta, ita_info,
                                          save_dir, k, ref_frenet, 4 * (r_n + 1)+1, col_id=(0, 1))
                    # swap the colors
                    rotate_crop_2path_fig(ref_paths, path1, path2, theta, ita_info,
                                          save_dir, k, ref_frenet, 4*(r_n+1)+2, col_id=(1, 0))
                    rotate_crop_2path_fig(ref_paths, path1, path2, theta, ita_info,
                                          save_dir, k, ref_frenet, 4 * (r_n + 1) + 3, col_id=(1, 0))
                count += (1+rotate_n)*4
                print(count)
    return


def get_ref_path(data, cx, cy, scene=''):
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
            if scene in ['FT', 'SR']:
                cp = cp[1:-1]
            ref_path_points[label] = np.vstack((v1['path'], cp, v2['path']))
    return ref_path_points


def get_circle(p1, p2, p3):
    x21 = p2[0] - p1[0]
    y21 = p2[1] - p1[1]
    x32 = p3[0] - p2[0]
    y32 = p3[1] - p2[1]
    # three colinear
    # if x21 * y32 - x32 * y21 == 0:
    #     return None
    xy21 = p2[0] * p2[0] - p1[0] * p1[0] + p2[1] * p2[1] - p1[1] * p1[1]
    xy32 = p3[0] * p3[0] - p2[0] * p2[0] + p3[1] * p3[1] - p2[1] * p2[1]
    y0 = (x32 * xy21 - x21 * xy32) / 2 * (y21 * x32 - y32 * x21)
    x0 = (xy21 - 2 * y0 * y21) / (2.0 * x21)
    r = ((p1[0] - x0) ** 2 + (p1[1] - y0) ** 2) ** 0.5
    return x0, y0, r


def fix_ref_path(ref_path_points, scene=''):
    if scene == 'FT':
        ref_path_points['1-10'][161] = (ref_path_points['1-10'][160] + ref_path_points['1-10'][162]) / 2
        ref_path_points['1-12'][136] = (ref_path_points['1-12'][135] + ref_path_points['1-12'][137])/2
    c_insert_k = 10
    l_insert_k = 2
    for k, v in ref_path_points.items():
        i = 1
        while i < len(v)-1:
            pre = v[i] - v[i-1]
            post = v[i+1] - v[i]
            theta1 = math.atan2(pre[1], pre[0])
            theta2 = math.atan2(post[1], pre[0])
            theta = ((theta2 - theta1)*180/math.pi)
            if theta > 180:
                theta -= 360
            elif theta < -180:
                theta += 360
            if abs(theta) > 45:
                print(theta)
                x0, y0, r = get_circle(v[i-1], v[i], v[i+1])
                theta_p = math.atan2(v[i-1][1]-y0, v[i-1][0]-x0)
                if theta_p < 0:
                    theta_p += math.pi * 2
                theta_c = math.atan2(v[i][1]-y0, v[i][0]-x0)
                if theta_c < 0:
                    theta_c += math.pi * 2
                p1 = []
                for j in range(1, c_insert_k):
                    x_rot, y_rot = counterclockwise_rotate(v[i - 1][0], v[i - 1][1], (x0, y0),
                                                           (theta_c - theta_p) * j / c_insert_k)
                    p1.append([x_rot, y_rot])
                p1 = np.array(p1)

                theta_n = math.atan2(v[i+1][1]-y0, v[i+1][0]-x0)
                if theta_n < 0:
                    theta_n += math.pi * 2
                p2 = []
                for j in range(1, c_insert_k):
                    x_rot, y_rot = counterclockwise_rotate(v[i-1][0], v[i-1][1], (x0, y0),
                                                           (theta_n-theta_c)*j/c_insert_k)
                    p2.append([x_rot, y_rot])
                p2 = np.array(p2)
                v = np.vstack((v[:i], p1, v[i], p2, v[i+1:]))
                i += c_insert_k * 2 - 1
            else:
                p1 = []
                for j in range(1, l_insert_k):
                    delta_x = (v[i][0]-v[i-1][0])/l_insert_k
                    delta_y = (v[i][1]-v[i-1][1])/l_insert_k
                    p1.append([v[i-1][0]+j*delta_x, v[i-1][1]+j*delta_y])
                p1 = np.array(p1)
                p2 = []
                for j in range(1, l_insert_k):
                    delta_x = (v[i+1][0]-v[i][0])/l_insert_k
                    delta_y = (v[i+1][1]-v[i][1])/l_insert_k
                    p2.append([v[i][0]+j*delta_x, v[i][1]+j*delta_y])
                p2 = np.array(p2)
                v = np.vstack((v[:i], p1, v[i], p2, v[i + 1:]))
                i += l_insert_k * 2
        ref_path_points[k] = v
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
    # judge if in some starting area in first 1/4 trajectory frame by frame
    start = 0
    for ts in range(track.time_stamp_ms_first, track.time_stamp_ms_last+100, 100):
        motion_state = track.motion_states[ts]
        cur_p = (motion_state.x, motion_state.y)
        for k, v in areas.items():
            this_start = judge_in_box(v['x'], v['y'], cur_p)

            if this_start == 1:
                if 'vel' in v.keys():
                    if 'y' in v['vel']:
                        vel = motion_state.vy * v['vel']['y']
                        if vel < 0:
                            continue
                    if 'x' in v['vel']:
                        vel = motion_state.vx * v['vel']['x']
                        if vel < 0:
                            continue
                start = k
                return start
    return start


def judge_end(track, areas):
    # judge if in some starting area frame by frame
    end = 0
    for ts in range(track.time_stamp_ms_first, track.time_stamp_ms_last+100, 100):
        motion_state = track.motion_states[ts]
        cur_p = (motion_state.x, motion_state.y)
        for k, v in areas.items():
            in_box = judge_in_box(v['x'], v['y'], cur_p)
            if in_box == 1:
                end = k
    return end


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


def get_track_label(dir_name, ref_path_points, ref_frenet, starting_areas, end_areas, scene=None):
    csv_dict = dict()
    # collect data to construct a dict from all csv
    paths = glob.glob(os.path.join(dir_name, '*.csv'))
    paths.sort()
    for csv_name in paths:
        print(csv_name)
        # if '025' not in csv_name:
        #     continue
        track_dictionary = dataset_reader.read_tracks(csv_name)
        tracks = dict_utils.get_value_list(track_dictionary)
        csv_agents = dict()
        for agent in tracks:
            # if agent.track_id != 171:
            #     continue
            start_area = judge_start(agent, starting_areas)
            end_area = judge_end(agent, end_areas)
            if start_area == 0 or end_area == 0:
                # print(agent.track_id, 'starting or ending area is 0, discard')
                continue
            path_name = str(start_area) + '-' + str(end_area)
            if path_name not in ref_path_points:
                path_name = str(start_area) + '--1-' + str(end_area)
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
            if scene == 'FT':
                if path_name == '1--1-2':
                    ms = agent.motion_states[end_ts]
                    x = ms.x
                    y = ms.y
                    closest_point_id = closest_point_index(x, y, xy_points)
                    if closest_point_id < len(xy_points)//2:
                        end_ts = end_ts - 1000
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
                    vs = (f_s - agent.motion_states[ts-100].frenet_s) / 0.1
                    agent.motion_states[ts].vs = vs
            if max_min_dis_traj > 100:
                continue
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
                agent_dict['motion_states'][ts]['proj'] = ms.proj
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


def save_all_edges(csv_dict, is_info, ref_frenet):
    all_edges = dict()
    for i, c_data in csv_dict.items():
        # get edge info in one csv file
        edges = get_csv_edges(c_data, is_info, ref_frenet, i)
        all_edges[i] = edges
        print(i, ': ', len(edges), '/', len(c_data))
    return all_edges


def save_per_ts_img(img_dir, img_path, ita1, ref_paths, path1, path2, theta1,
                    cls_id, ref_frenet, img_name1):
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
        print('make dir: ', img_dir)
    if not os.path.exists(img_path):
        rotate_crop_2path_fig(ref_paths, path1, path2, theta1, ita1, img_dir, cls_id,
                              ref_frenet, 0, img_name1, (0, 1))
    if not os.path.exists(img_path.replace('0.', '1.')):
        rotate_crop_2path_fig(ref_paths, path1, path2, theta1, ita1, img_dir, cls_id,
                              ref_frenet, 1, img_name1.replace('0.', '1.'), (0, 1))
    if not os.path.exists(img_path.replace('0.', '2.')):
        rotate_crop_2path_fig(ref_paths, path1, path2, theta1, ita1, img_dir, cls_id,
                              ref_frenet, 2, img_name1.replace('0.', '2.'), (1, 0))
    if not os.path.exists(img_path.replace('0.', '3.')):
        rotate_crop_2path_fig(ref_paths, path1, path2, theta1, ita1, img_dir, cls_id,
                              ref_frenet, 3, img_name1.replace('0.', '3.'), (1, 0))
        # ita2 = ita_info[path2][path1][cls_id]
        # theta2 = math.pi/2 - cur_ms2['psi_rad']
        # img_name2 = '{}_{}_{}_{}.png'.format(csv_key, id2, id1, cur_ts)
        # rotate_crop_2path_fig(ref_paths, path2, path1, theta2, ita2, img_dir,
        #                       cls_id, ref_frenet, 0, img_name2, col_id=(0, 1))
        # rotate_crop_2path_fig(ref_paths, path2, path1, theta2, ita2, img_dir,
        #                       cls_id, ref_frenet, 1, img_name2, col_id=(0, 1))
        # rotate_crop_2path_fig(ref_paths, path2, path1, theta2, ita2, img_dir,
        #                       cls_id, ref_frenet, 2, img_name2, col_id=(1, 0))
        # rotate_crop_2path_fig(ref_paths, path2, path1, theta2, ita2, img_dir,
        #                       cls_id, ref_frenet, 3, img_name2, col_id=(1, 0))


def get_csv_edges(c_data, ita_info, ref_frenet, csv_key, img_dir, ref_paths):
    edges = dict()
    for ego_id, ego_data in c_data.items():
        print(ego_id)
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
            edges[ego_id][start_ts]['ita_id'] = dict()
            case_ita_s = dict()
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
                    # have interaction and not arrived
                    if path2 in ita_info[path1].keys():
                        # judge if having passed the intersection
                        interactions = ita_info[path1][path2]
                        closest_k = -1
                        closest_dis = 1e10
                        ms2_20 = c_data[id2]['motion_states'][ts_20]
                        # find the nearest intersection of agent1 and agent2
                        for content_k, content in enumerate(interactions):
                            p = content[0]
                            dis = (p[0] - ms1_20['x']) ** 2 + (p[1] - ms1_20['y']) ** 2
                            dis += (p[0] - ms2_20['x']) ** 2 + (p[1] - ms2_20['y']) ** 2
                            if dis < closest_dis:
                                closest_dis = dis
                                closest_k = content_k
                        closest_its = interactions[closest_k]
                        ita_s1 = ref_frenet[path1][closest_its[1]]
                        ita_s2 = ref_frenet[path2][closest_its[2]]
                        # having passed the intersection in the 1st frame, no interaction
                        if ms1_start['frenet_s'] > ita_s1 and ms2_start['frenet_s'] > ita_s2:
                            continue
                        interaction_num += 1
                        if id1 not in edges[ego_id][start_ts]['ita_id'].keys():
                            edges[ego_id][start_ts]['ita_id'][id1] = dict()
                        edges[ego_id][start_ts]['ita_id'][id1][id2] = closest_k
                        if id2 not in edges[ego_id][start_ts]['ita_id'].keys():
                            edges[ego_id][start_ts]['ita_id'][id2] = dict()
                        edges[ego_id][start_ts]['ita_id'][id2][id1] = closest_k
                        if id1 not in case_ita_s.keys():
                            case_ita_s[id1] = dict()
                        case_ita_s[id1][id2] = (ita_s1, ita_s2)
                        if id2 not in case_ita_s.keys():
                            case_ita_s[id2] = dict()
                        case_ita_s[id2][id1] = (ita_s2, ita_s1)
                        pair = sorted([path1, path2])
                        task = pair[0] + '_' + pair[1] + '_' + str(closest_k)
                        edges[ego_id][start_ts]['task'].add(task)
                    # in the same ref path and ego behind other
                    elif path2 == path1:
                        interaction_num += 1
                        if id1 not in edges[ego_id][start_ts]['ita_id'].keys():
                            edges[ego_id][start_ts]['ita_id'][id1] = dict()
                        edges[ego_id][start_ts]['ita_id'][id1][id2] = 0
                        if id2 not in edges[ego_id][start_ts]['ita_id'].keys():
                            edges[ego_id][start_ts]['ita_id'][id2] = dict()
                        edges[ego_id][start_ts]['ita_id'][id2][id1] = 0
                        task = path1 + '_' + path2 + '_0'
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
            # for agent_id in edges[ego_id][start_ts]['agents']:
            #     edges[ego_id][start_ts]['xy'][agent_id] = get_70_coor(c_data[agent_id], start_ts)

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

                        # save interaction info of car1 and car2
                        if path1 == path2:
                            rel_x, rel_y = cur_ms2['x']-cur_ms1['x'], cur_ms2['y']-cur_ms1['y']
                            rot_x, rot_y = counterclockwise_rotate(rel_x, rel_y, (0, 0), theta1)
                            delta_s = cur_ms2['frenet_s'] - cur_ms1['frenet_s']
                            edges[ego_id][start_ts][cur_ts][id1][id2] = [(rot_x, rot_y), delta_s]

                        elif id1 in case_ita_s.keys() and id2 in case_ita_s[id1].keys():
                            rel_x, rel_y = cur_ms2['x'] - cur_ms1['x'], cur_ms2['y'] - cur_ms1['y']
                            rot_x, rot_y = counterclockwise_rotate(rel_x, rel_y, (0, 0), theta1)
                            ita_s1, ita_s2 = case_ita_s[id1][id2]
                            s_1 = ita_s1 - cur_ms1['frenet_s']
                            s_2 = ita_s2 - cur_ms2['frenet_s']
                            edges[ego_id][start_ts][cur_ts][id1][id2] = [(rot_x, rot_y), s_1, s_2, theta1]
                            # cls_id = edges[ego_id][start_ts]['ita_id'][id1][id2]
                            # ita1 = ita_info[path1][path2][cls_id]
                            # img_name1 = '{}_{}_{}_{}_{}_0.png'.format(csv_key, id1, id2, cur_ts, cls_id)
                            # img_path = img_dir + img_name1
                            # save_per_ts_img(img_dir, img_path, ita1, ref_paths, path1, path2,
                            #                 theta1, cls_id, ref_frenet, img_name1)
        if ego_id in edges.keys() and len(edges[ego_id]) == 0:
            del edges[ego_id]
    return edges


def plot_csv_imgs(edges, c_data, ita_info, ref_frenet, csv_key, img_dir, ref_paths):
    for ego_id, e1_data in edges.items():
        print(ego_id)
        for start_ts, e2_data in e1_data.items():
            for cur_ts, e3_data in e2_data.items():
                if not isinstance(cur_ts, int):
                    continue
                for id1, e4_data in e3_data.items():
                    if e4_data is None:
                        continue
                    for id2 in e4_data.keys():
                        cur_ms1 = c_data[id1]['motion_states'][cur_ts]
                        path1 = c_data[id1]['ref path']
                        theta1 = math.pi / 2 - cur_ms1['psi_rad']
                        path2 = c_data[id2]['ref path']
                        if path1 == path2:
                            continue
                        cls_id = edges[ego_id][start_ts]['ita_id'][id1][id2]
                        ita1 = ita_info[path1][path2][cls_id]
                        img_name1 = '{}_{}_{}_{}_{}_0.png'.format(csv_key, id1, id2, cur_ts, cls_id)
                        img_path = img_dir + img_name1
                        save_per_ts_img(img_dir, img_path, ita1, ref_paths, path1, path2,
                                        0, cls_id, ref_frenet, img_name1)


def save_ts_theta(csv_data, save_path):
    data = dict()
    for k, c_data in csv_data.items():
        data[k] = dict()
        for ego_id, ego_data in c_data.items():
            data[k][ego_id] = dict()
            for ts in range(ego_data['time_stamp_ms_first'], ego_data['time_stamp_ms_last'] + 100, 100):
                ego_ms = ego_data['motion_states'][ts]
                data[k][ego_id][ts] = [ego_ms['x'], ego_ms['y'], ego_ms['psi_rad']]
    pickle_file = open(save_path, 'wb')
    pickle.dump(data, pickle_file)
    pickle_file.close()
    return data


def rotate_crop_ts(img_path, data, xs, ys):
    import matplotlib.pyplot as plt
    import math
    x, y, psi_rad = data
    theta = math.pi/2 - psi_rad
    from PIL import Image
    img = Image.open(img_path)
    w, h = img.size
    xc = int(1 + (x - xs) * 1.6)
    yc = h - int(1 + (y - ys) * 1.6)
    img = img.rotate(theta*180/math.pi, center=(xc, yc), resample=Image.BILINEAR)
    img = img.crop((xc-16, yc-16, xc+16, yc+16))
    print(img.size, xc, yc)
    plt.figure()
    plt.imshow(img)
    plt.show()


# if __name__ == '__main__':
#     dataset = 'SR'
#
#     import pickle
#     # path
#     pickle_file = open('D:/Dev/UCB task/pickle/{}/ts_theta_{}.pkl'.format(dataset, dataset), 'rb')
#     data = pickle.load(pickle_file)
#     pickle_file.close()
#
#     start_xy = {'SR': (900, 965), 'FT': (945, 945), 'MA': (955, 945)}
#     xs, ys = start_xy[dataset]
#     # in 000.csv, for the car id=11 whose ref path='5--1-2', in time step=fts, get the 32*32 image
#     # data[csv_id][car_id][time step] return [x, y, psi_rad]
#     path = '5--1-2'
#     fts = 3600
#     rotate_crop_ts('D:/Dev/UCB task/intersection_figs/single_SR/{}.png'.format(path), data['000'][11][fts], xs, ys)
