import scipy.io as scio
import pickle
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
FT_starting_area_dict[11]['x'] = [1033.5, 1037, 1041.6, 1038]
FT_starting_area_dict[11]['y'] = [977.1, 972.3, 973.7, 978.7]
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
    # a dict, call by path and return an array(x,2)
    FT_ref_path_points = get_ref_path(data['Segmented_reference_path'], cx, cy, scene='FT')

    # complete missing trajectory to make paths have the same frenet starting point
    complete = [['1-10', '1--1-2', '3--1-10'], ['1-12', '1--1-2', '3--1-12'],
                ['3-2', '3--1-4', '1--1-2'], ['7-4', '7--1-2', '1--1-4'],
                ['7-6', '7--1-2', '1--1-6'], ['9-6', '9--1-2', '1--1-6'],
                ['9-8', '9--1-2', '1--1-8'], ['11-6', '11--1-2', '1--1-6'],
                ['11-8', '11--1-2', '1--1-8']]
    for cmp, ref_pre, ref_post in complete:
        xp1 = FT_ref_path_points[cmp][:, 0]
        yp1 = FT_ref_path_points[cmp][:, 1]
        xp2 = FT_ref_path_points[ref_pre][:, 0]
        yp2 = FT_ref_path_points[ref_pre][:, 1]
        xp3 = FT_ref_path_points[ref_post][:, 0]
        yp3 = FT_ref_path_points[ref_post][:, 1]
        pre_x, pre_y = get_append(xp1[0], yp1[0], xp2, yp2, 'start')
        post_x, post_y = get_append(xp1[-1], yp1[-1], xp3, yp3, 'end')
        xp1 = pre_x[:-1] + list(xp1) + post_x[1:]
        yp1 = pre_y[:-1] + list(yp1) + post_y[1:]
        xyp1 = np.array([[x1, y1] for x1, y1 in zip(xp1, yp1)])
        FT_ref_path_points[cmp] = xyp1
    insert_k = 10
    FT_ref_path_points = fix_ref_path(FT_ref_path_points, 'FT', insert_k=insert_k)
    plot_ref_path(map_dir + map_name, FT_ref_path_points, FT_starting_area_dict, FT_end_area_dict)
    # pickle_file = open('D:/Dev/UCB task/pickle/FT/ref_path_xy_FT.pkl', 'wb')
    # pickle.dump(FT_ref_path_points, pickle_file)
    # pickle_file.close()
    # a dict, call by path return an array(x,1): frenet of ref path points
    # ref_point_frenet = ref_paths2frenet(FT_ref_path_points)
    # pickle_file = open('D:/Dev/UCB task/pickle/FT/ref_path_frenet_FT.pkl', 'wb')
    # pickle.dump(ref_point_frenet, pickle_file)
    # pickle_file.close()
    # FT_interactions = find_all_interactions(FT_ref_path_points, k=insert_k)
    # pickle_file = open('D:/Dev/UCB task/pickle/FT/interaction_FT.pkl', 'wb')
    # pickle.dump(FT_interactions, pickle_file)
    # pickle_file.close()
    # save_interaction_bg_figs(FT_ref_path_points, FT_interactions, map_dir+map_name,
    #                          'D:/Dev/UCB task/intersection_figs/roundabout_FT/')

    img_save_dir = 'D:/Dev/UCB task/intersection_figs/high-res_roundabout_FT_crop/'
    rotate_n = 0
    # crop_interaction_figs(FT_ref_path_points, FT_interactions, ref_point_frenet, img_save_dir, rotate_n)
    # save_complete_ref_path_fig(FT_ref_path_points, 'D:/Dev/UCB task/intersection_figs/single_FT/',
    #                            (945, 1070), (945, 1050))

    # if not os.path.exists('D:/Dev/UCB task/pickle/FT/track_path_frenet_FT.pkl'):
    #     pickle_file = open('D:/Dev/UCB task/pickle/FT/track_path_frenet_FT.pkl', 'rb')
    #     csv_data = pickle.load(pickle_file)
    #     pickle_file.close()
    # else:
    #     csv_data = get_track_label('D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/DR_USA_Roundabout_FT/',
    #                                FT_ref_path_points, ref_point_frenet,
    #                                FT_starting_area_dict, FT_end_area_dict, 'FT')
    #     pickle_file = open('D:/Dev/UCB task/pickle/FT/track_path_frenet_FT.pkl', 'wb')
    #     pickle.dump(csv_data, pickle_file)
    #     pickle_file.close()
    # save_ts_theta(csv_data, 'D:/Dev/UCB task/pickle/FT/ts_theta_FT.pkl')
    # for k, v in csv_data.items():
    #     print(k)
    #     split_edges = get_csv_edges(v, FT_interactions, ref_point_frenet, k,
    #                                 img_save_dir+k+'/', FT_ref_path_points)
    #     pickle_file = open('D:/Dev/UCB task/pickle/FT/edges_FT_{}.pkl'.format(k), 'wb')
    #     pickle.dump(split_edges, pickle_file)
    #     pickle_file.close()
