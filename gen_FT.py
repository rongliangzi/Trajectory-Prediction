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
from utils.roundabout_utils import *

FT_starting_area_dict = dict()
FT_starting_area_dict[1] = dict()
FT_starting_area_dict[1]['x'] = [990, 990.2, 1000.3, 1000]
FT_starting_area_dict[1]['y'] = [988, 984, 983.5, 988]
FT_starting_area_dict[3] = dict()
FT_starting_area_dict[3]['x'] = [994.5, 999, 1004.5, 1002.1]
FT_starting_area_dict[3]['y'] = [1019, 1010, 1011.9, 1020.2]
FT_starting_area_dict[5] = dict()
FT_starting_area_dict[5]['x'] = [1036, 1030, 1032.5, 1039]
FT_starting_area_dict[5]['y'] = [1024, 1019, 1016, 1023]
FT_starting_area_dict[7] = dict()
FT_starting_area_dict[7]['x'] = [1041, 1041.2, 1051, 1050.5]
FT_starting_area_dict[7]['y'] = [1010, 1005.5, 1007.4, 1010.5]
FT_starting_area_dict[9] = dict()
FT_starting_area_dict[9]['x'] = [1040.5, 1045.2, 1047.4, 1041.6]
FT_starting_area_dict[9]['y'] = [984.3, 981.2, 984.6, 989.2]
FT_starting_area_dict[11] = dict()
FT_starting_area_dict[11]['x'] = [1033, 1038, 1041.3, 1037.1]
FT_starting_area_dict[11]['y'] = [978, 971, 973, 980]

FT_end_area_dict = dict()
FT_end_area_dict[2] = dict()
FT_end_area_dict[2]['x'] = [991.5, 985.4, 985.9, 992.2]
FT_end_area_dict[2]['y'] = [999, 996.2, 993.7, 995.4]
FT_end_area_dict[4] = dict()
FT_end_area_dict[4]['x'] = [1003.5, 1007.1, 1012.2, 1009.6]
FT_end_area_dict[4]['y'] = [1022.8, 1019.5, 1023.2, 1026.3]
FT_end_area_dict[6] = dict()
FT_end_area_dict[6]['x'] = [1035.7, 1040.8, 1044.8, 1040.3]
FT_end_area_dict[6]['y'] = [1015.4, 1012.3, 1018.1, 1021.9]
FT_end_area_dict[8] = dict()
FT_end_area_dict[8]['x'] = [1042, 1043.8, 1049.3, 1048.6]
FT_end_area_dict[8]['y'] = [1004.8, 1001.5, 1003.4, 1006]
FT_end_area_dict[10] = dict()
FT_end_area_dict[10]['x'] = [1041.6, 1049.3, 1051.3, 1043.6]
FT_end_area_dict[10]['y'] = [978.6, 973.4, 976.5, 981.9]
FT_end_area_dict[12] = dict()
FT_end_area_dict[12]['x'] = [1029.2, 1034.4, 1038.6, 1033.2]
FT_end_area_dict[12]['y'] = [975.4, 967.6, 969.3, 977]


def plot_start_end_area(ax):
    for key, v in FT_starting_area_dict.items():
        x = v['x']
        y = v['y']
        ax.text(x[0], y[0], key, fontsize=20)
        ax.plot(x[0:2], y[0:2], c='r', zorder=40)
        ax.plot(x[1:3], y[1:3], c='r', zorder=40)
        ax.plot(x[2:4], y[2:4], c='r', zorder=40)
        ax.plot(x[3:] + x[0:1], y[3:] + y[0:1], c='r', zorder=40)
    for key, v in FT_end_area_dict.items():
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
    plot_start_end_area(axes)
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
    plt.show()


if __name__ == '__main__':
    map_dir = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/'
    map_name = "DR_USA_Roundabout_FT.osm"
    dataFile = 'D:/Dev/UCB task/Segmented_reference_path_DR_USA_Roundabout_FT.mat'
    data = scio.loadmat(dataFile)
    para_path = data['Segmented_reference_path']['para_path']
    circle_merge_point = data['Segmented_reference_path']['circle_merge_point'][0]
    # plot_raw_ref_path(map_dir + map_name, para_path, circle_merge_point)

    cx, cy = fit_circle(circle_merge_point)
    # a dict, call by path return an array(x,2)
    FT_ref_path_points = get_ref_path(data['Segmented_reference_path'], cx, cy)

    # plot_ref_path_divided(map_dir + map_name, ref_path_points)
    # plot_ref_path(map_dir + map_name, FT_ref_path_points)
    FT_intersections = find_all_intersections(FT_ref_path_points)
    # save_intersection_bg_figs(FT_ref_path_points, FT_intersections, map_dir+map_name,
    #                           'D:/Dev/UCB task/intersection_figs/roundabout_FT/')
    # a dict, call by path return an array(x,1): frenet of ref path points
    ref_point_frenet = ref_paths2frenet(FT_ref_path_points)
    crop_intersection_figs(FT_ref_path_points, FT_intersections, ref_point_frenet,
                           'D:/Dev/UCB task/intersection_figs/roundabout_FT_crop/')

    # csv_data = get_track_label('D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/DR_USA_Roundabout_FT/',
    #                            FT_ref_path_points, ref_point_frenet, FT_starting_area_dict, FT_end_area_dict)
    # pickle_file = open('D:/Dev/UCB task/pickle/track_path_frenet_FT.pkl', 'wb')
    # pickle.dump(csv_data, pickle_file)
    # pickle_file.close()
