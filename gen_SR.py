import matplotlib.patches as patches
import scipy.io as scio
import numpy as np
import math
import glob
import os
import pickle

import matplotlib.pyplot as plt
from utils import map_vis_without_lanelet
from utils.new_coor_ref_path_utils import judge_in_box


from utils.roundabout_utils import *

SR_starting_area_dict = dict()
SR_starting_area_dict[1] = dict()
SR_starting_area_dict[1]['x'] = [963.4, 964.1, 972.2, 971.7]
SR_starting_area_dict[1]['y'] = [1014.6, 1006.9, 1005.9, 1013.4]
SR_starting_area_dict[1]['stopline'] = [(969.7, 1013.4), (970.2, 1005.9)]
SR_starting_area_dict[3] = dict()
SR_starting_area_dict[3]['x'] = [984.3, 979.1, 989.1, 991.15]
SR_starting_area_dict[3]['y'] = [1053, 1045.7, 1044.5, 1052.5, ]
SR_starting_area_dict[3]['stopline'] = [(980.5, 1047.3), (989.1, 1046.5)]
SR_starting_area_dict[5] = dict()
SR_starting_area_dict[5]['x'] = [1008.7, 1008.5, 1016.5, 1016.6]
SR_starting_area_dict[5]['y'] = [1034.7, 1027.8, 1026.4, 1033.2]
SR_starting_area_dict[5]['stopline'] = [(1010.7, 1034.7), (1010.5, 1027.8)]
SR_starting_area_dict[7] = dict()
SR_starting_area_dict[7]['x'] = [988.5, 988.5, 992.8, 995.6]
SR_starting_area_dict[7]['y'] = [991.4, 983.4, 983, 990.5]
SR_starting_area_dict[7]['stopline'] = [(988.5, 989.4), (995.6, 988.5)]

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


def plot_start_end_area(ax):
    for key, v in SR_starting_area_dict.items():
        x = v['x']
        y = v['y']
        stopline = v['stopline']
        stop_x = [st[0] for st in stopline]
        stop_y = [st[1] for st in stopline]
        ax.plot(stop_x, stop_y, c='g', linewidth=5, zorder=30)
        ax.text(x[0], y[0], key, fontsize=20)
        ax.plot(x[0:2], y[0:2], c='r', zorder=40)
        ax.plot(x[1:3], y[1:3], c='r', zorder=40)
        ax.plot(x[2:4], y[2:4], c='r', zorder=40)
        ax.plot(x[3:] + x[0:1], y[3:] + y[0:1], c='r', zorder=40)
    for key, v in SR_end_area_dict.items():
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
    plot_start_end_area(axes)
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
    plot_start_end_area(axes)
    fig.canvas.mpl_connect('button_press_event', on_press)
    plt.show()


if __name__ == '__main__':
    map_dir = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/'
    map_name = "DR_USA_Roundabout_SR.osm"
    dataFile = 'D:/Dev/UCB task/Segmented_reference_path_DR_USA_Roundabout_SR.mat'
    mat_data = scio.loadmat(dataFile)
    mat_data = mat_data['Segmented_reference_path']
    all_circle_points = mat_data['circle_merge_point'][0]
    # plot_raw_ref_path(map_dir + map_name, mat_data['para_path'], circle_merge_point)

    circle_x, circle_y = fit_circle(all_circle_points)
    # a dict, call by path return an array(x,2)
    SR_ref_path_points = get_ref_path(mat_data, circle_x, circle_y)
    # a dict, call by path return an array(x,1): frenet of ref path points
    ref_point_frenet = ref_paths2frenet(SR_ref_path_points)
    SR_intersections = find_all_intersections(SR_ref_path_points)
    SR_split = find_all_split_points(SR_ref_path_points)
    # save_intersection_bg_figs(SR_ref_path_points, SR_intersections, map_dir+map_name,
    #                           'D:/Dev/UCB task/intersection_figs/roundabout_SR/')
    # rotate_n = 49
    # crop_intersection_figs(SR_ref_path_points, SR_intersections, ref_point_frenet,
    #                        'D:/Dev/UCB task/intersection_figs/roundabout_SR_crop/', rotate_n)
    # crop_split_figs(SR_ref_path_points, SR_split, ref_point_frenet,
    #                 'D:/Dev/UCB task/intersection_figs/roundabout_SR_crop/', rotate_n)
    plot_ref_path(map_dir + map_name, SR_ref_path_points)
    # if os.path.exists('D:/Dev/UCB task/pickle/track_path_frenet_SR.pkl'):
    #     pickle_file = open('D:/Dev/UCB task/pickle/track_path_frenet_SR.pkl', 'rb')
    #     csv_data = pickle.load(pickle_file)
    #     pickle_file.close()
    # else:
    #     csv_data = get_track_label('D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/DR_USA_Roundabout_SR/',
    #                                SR_ref_path_points, ref_point_frenet,
    #                                SR_starting_area_dict, SR_end_area_dict)
    #     pickle_file = open('D:/Dev/UCB task/pickle/track_path_frenet_SR.pkl', 'wb')
    #     pickle.dump(csv_data, pickle_file)
    #     pickle_file.close()
    # all_edges = save_edges(csv_data, SR_intersections, ref_point_frenet, SR_starting_area_dict)
    #
    # pickle_file = open('D:/Dev/UCB task/pickle/edges_SR.pkl', 'wb')
    # pickle.dump(all_edges, pickle_file)
    # pickle_file.close()

