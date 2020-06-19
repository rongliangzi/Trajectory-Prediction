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
FT_starting_area_dict[1]['x'] = [987, 988.2, 996.2, 995]
FT_starting_area_dict[1]['y'] = [988, 984, 984, 988]
FT_starting_area_dict[1]['stopline'] = [(993, 988), (994.2, 984)]
FT_starting_area_dict[3] = dict()
FT_starting_area_dict[3]['x'] = [992.7, 997.5, 1003.3, 998.8]
FT_starting_area_dict[3]['y'] = [1020.6, 1014.4, 1018.1, 1025]
FT_starting_area_dict[3]['stopline'] = [(996.7, 1015.9), (1002.2, 1019.7)]
FT_starting_area_dict[5] = dict()
FT_starting_area_dict[5]['x'] = [1032.2, 1034.7, 1039.9, 1037]
FT_starting_area_dict[5]['y'] = [1019.9, 1017.8, 1024, 1026.3]
FT_starting_area_dict[5]['stopline'] = [(1033.8, 1021.1), (1036.4, 1019)]
FT_starting_area_dict[7] = dict()
FT_starting_area_dict[7]['x'] = [1043.3, 1044.6, 1052.4, 1051]
FT_starting_area_dict[7]['y'] = [1009.2, 1005.5, 1007.5, 1011.7]
FT_starting_area_dict[7]['stopline'] = [(1045.5, 1009.4), (1046.6, 1006)]
FT_starting_area_dict[9] = dict()
FT_starting_area_dict[9]['x'] = [1041.5, 1048.4, 1050, 1043.2]
FT_starting_area_dict[9]['y'] = [983.6, 979, 983.2, 988.2]
FT_starting_area_dict[9]['stopline'] = [(1043.5, 982.6), (1044.9, 986.5)]
FT_starting_area_dict[11] = dict()
FT_starting_area_dict[11]['x'] = [1032.2, 1037, 1041.6, 1037]
FT_starting_area_dict[11]['y'] = [978.9, 972.3, 973.7, 980.4]
FT_starting_area_dict[11]['stopline'] = [(1033.7, 977), (1037.9, 978.5)]

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
    plot_ref_path(map_dir + map_name, FT_ref_path_points, FT_starting_area_dict, FT_end_area_dict)
    FT_intersections = find_all_intersections(FT_ref_path_points)
    FT_split_points = find_all_split_points(FT_ref_path_points)
    # save_intersection_bg_figs(FT_ref_path_points, FT_intersections, map_dir+map_name,
    #                           'D:/Dev/UCB task/intersection_figs/roundabout_FT/')
    # a dict, call by path return an array(x,1): frenet of ref path points
    ref_point_frenet = ref_paths2frenet(FT_ref_path_points)
    rotate_n = 49
    # crop_intersection_figs(FT_ref_path_points, FT_intersections, ref_point_frenet,
    #                        'D:/Dev/UCB task/intersection_figs/roundabout_FT_crop/', rotate_n)
    # crop_split_figs(FT_ref_path_points, FT_split_points, ref_point_frenet,
    #                 'D:/Dev/UCB task/intersection_figs/roundabout_FT_crop/', rotate_n)
    # if os.path.exists('D:/Dev/UCB task/pickle/track_path_frenet_FT.pkl'):
    #     pickle_file = open('D:/Dev/UCB task/pickle/track_path_frenet_FT.pkl', 'rb')
    #     csv_data = pickle.load(pickle_file)
    #     pickle_file.close()
    # else:
    #     csv_data = get_track_label('D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/DR_USA_Roundabout_FT/',
    #                                FT_ref_path_points, ref_point_frenet, FT_starting_area_dict, FT_end_area_dict)
    #     pickle_file = open('D:/Dev/UCB task/pickle/track_path_frenet_FT.pkl', 'wb')
    #     pickle.dump(csv_data, pickle_file)
    #     pickle_file.close()
    # edge_keys = sorted(csv_data.keys())
    # s = 4
    # for k in range(s):
    #     start = k*len(edge_keys)//s
    #     end = min((k+1)*len(edge_keys)//s, len(edge_keys))
    #     split_data = dict()
    #     for ek in range(start, end):
    #         split_data[edge_keys[ek]] = csv_data[edge_keys[ek]]
    #     split_edges = save_edges(split_data, FT_intersections, ref_point_frenet,
    #                              FT_starting_area_dict, FT_split_points)
    #     pickle_file = open('D:/Dev/UCB task/pickle/edges_FT_{}.pkl'.format(k), 'wb')
    #     pickle.dump(split_edges, pickle_file)
    #     pickle_file.close()
