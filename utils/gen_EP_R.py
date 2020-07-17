from utils.EP_R_utils import *
from utils.EP_T_utils import get_track_label
from utils.roundabout_utils import fix_ref_path, save_interaction_bg_figs, save_complete_ref_path_fig,\
    crop_interaction_figs, ref_paths2frenet, save_ts_theta, get_csv_edges
import pickle


if __name__ == '__main__':
    func_file = 'D:/Dev/UCB task/Roundabout_EP_final/ref_path_funcs.txt'
    map_path = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/DR_USA_Roundabout_EP.osm'
    EPR_ref_path_points = read_funcs(func_file)
    # insert_k = 2
    # EPR_ref_path_points = fix_ref_path(EPR_ref_path_points, 'EPR', insert_k=insert_k)
    EPR_interaction_points = {'1-4_1-6': (((976.25, 1013.14), 'split_0'),),
                              '1-4_1-9': (((976.25, 1013.14), 'split_0'),),
                              '1-4_10-2': (((970.28, 1034.54), 'crossing_0'),),
                              '1-4_10-4': (((976.16, 1012.87), 'merging_0'),),
                              '1-4_10-6': (((976.23, 1013.01), 'crossing_0'),),
                              '1-4_3-4': (((971.64, 1028.24), 'merging_0'),),
                              '1-4_3-6': (((971.64, 1028.18), 'merging_0'), ((976.22, 1013.31), 'split_0')),
                              '1-4_3-9': (((971.64, 1028.18), 'merging_0'), ((976.22, 1013.31), 'split_0')),
                              '1-4_5-0': (((968.75, 1006.37), 'crossing_0'),),
                              '1-4_5-2': (((968.75, 1006.37), 'crossing_0'), ((970.38, 1034.73), 'crossing_1')),
                              '1-4_5-9': (((968.75, 1006.37), 'crossing_0'),),
                              '1-4_7-2': (((970.24, 1034.44), 'crossing_0'),),
                              '1-4_7-4': (((976.01, 1013.52), 'merging_0'),),
                              '1-6_1-9': (((976.88, 1009.77), 'split_0'),),
                              '1-6_10-2': (((970.28, 1034.54), 'crossing_0'),),
                              '1-6_10-4': (((976.31, 1012.98), 'crossing_0'),),
                              '1-6_10-6': (((976.32, 1012.92), 'crossing_0'), ((974.24, 990.91), 'merging_0')),
                              '1-6_3-4': (((971.61, 1028.26), 'merging_0'), ((976.25, 1012.98), 'split_0')),
                              '1-6_3-6': (((971.25, 1028.67), 'merging_0'),),
                              '1-6_3-9': (((971.24, 1028.53), 'merging_0'), ((976.73, 1009.63), 'split_0')),
                              '1-6_5-0': (((976.88, 1001.98), 'crossing_0'),),
                              '1-6_5-2': (((976.87, 1002.12), 'crossing_0'), ((970.38, 1034.73), 'crossing_1')),
                              '1-6_5-6': (((974.28, 991.88), 'merging_0'),),
                              '1-6_5-9': (((976.88, 1001.98), 'crossing_0'),),
                              '1-6_7-2': (((970.09, 1034.59), 'crossing_0'),),
                              '1-6_7-4': (((976.15, 1013.09), 'crossing_0'),),
                              '1-9_10-2': (((970.23, 1034.73), 'crossing_0'),),
                              '1-9_10-4': (((976.15, 1012.94), 'crossing_0'),),
                              '1-9_10-6': (((976.15, 1012.94), 'crossing_0'),),
                              '1-9_3-4': (((971.61, 1028.26), 'merging_0'), ((976.25, 1012.98), 'split_0')),
                              '1-9_3-6': (((971.1, 1028.96), 'merging_0'), ((976.87, 1009.48), 'split_0')),
                              '1-9_3-9': (((971.1, 1028.96), 'merging_0'), ),
                              '1-9_5-0': (((982.21, 1001.69), 'merging_0'), ((993.6, 1004.6), 'split_0')),
                              '1-9_5-2': (((982.1, 1001.5), 'merging_0'), ((993.7, 1004.6), 'split_0'),
                                          ((970.1, 1034.7), 'crossing_0')),
                              '1-9_5-9': (((981.9, 1001.5), 'merging_0'),),
                              '1-9_7-0': (((993.6, 1004.4), 'crossing_0'),),
                              '1-9_7-2': (((970.2, 1034.4), 'crossing_0'), ((993.5, 1004.3), 'crossing_1')),
                              '1-9_7-4': (((976.3, 1012.9), 'crossing_0'), ((993.5, 1004.3), 'crossing_1')),
                              '1-9_7-9': (((999.4, 1006.8), 'merging_0'),),
                              '1-9_8-9': (((1002.6, 1008.3), 'merging_0'),),
                              '10-0_10-2': (((984, 1025.1), 'split_0'),),
                              '10-0_10-4': (((998.3, 1018.7), 'split_0'),),
                              '10-0_10-6': (((998.3, 1018.7), 'split_0'),),
                              '10-0_5-0': (((984.2, 1024.9), 'merging_0'),),
                              '10-0_5-2': (((984.3, 1024.9), 'crossing_0'),),
                              '10-0_7-0': (((984.2, 1024.9), 'merging_0'),),
                              '10-0_7-2': (((984.3, 1024.9), 'crossing_0'),),
                              '10-2_10-4': (((998.3, 1018.7), 'split_0'),),
                              '10-2_10-6': (((998.3, 1018.7), 'split_0'),),
                              '10-2_5-0': (((984, 1025), 'crossing_0'),),
                              '10-2_5-2': (((984, 1025), 'merging_0'),),
                              '10-2_7-0': (((984, 1025), 'crossing_0'),),
                              '10-2_7-2': (((984, 1025), 'merging_0'),),
                              '10-4_10-6': (((976.3, 1013), 'split_0'),),
                              '10-4_3-4': (((976.2, 1012.9), 'merging_0'),),
                              '10-4_3-6': (((976.2, 1012.9), 'crossing_0'),),
                              '10-4_3-9': (((976.2, 1012.9), 'crossing_0'),),
                              '10-4_5-0': (((990.6, 1019.7), 'crossing_0'), ((968.8, 1006.4), 'crossing_1')),
                              '10-4_5-2': (((990.6, 1019.7), 'crossing_0'), ((968.8, 1006.4), 'crossing_1')),
                              '10-4_5-9': (((968.8, 1006.4), 'crossing_0'),),
                              '10-4_7-0': (((990.6, 1019.8), 'crossing_0'),),
                              '10-4_7-2': (((990.6, 1019.8), 'crossing_0'),),
                              '10-4_7-4': (((990.6, 1019.8), 'merging_0'),),
                              '10-6_3-4': (((976.3, 1013), 'crossing_0'),),
                              '10-6_3-6': (((976.32, 1012.92), 'crossing_0'), ((974.24, 990.91), 'merging_0')),
                              '10-6_3-9': (((976.3, 1013), 'crossing_0'),),
                              '10-6_5-0': (((990.6, 1019.7), 'crossing_0'), ((975.5, 1002.4), 'crossing_1')),
                              '10-6_5-2': (((990.6, 1019.7), 'crossing_0'), ((975.5, 1002.4), 'crossing_1')),
                              '10-6_5-6': (((974.4, 992.1), 'merging_0'),),
                              '10-6_5-9': (((975.4, 1002.4), 'crossing_0'),),
                              '10-6_7-0': (((990.5, 1019.8), 'crossing_0'),),
                              '10-6_7-2': (((990.5, 1019.8), 'crossing_0'),),
                              '10-6_7-4': (((990.6, 1019.8), 'merging_0'), ((976.3, 1012.9), 'split_0')),
                              '3-4_3-6': (((976.3, 1013.1), 'split_0'),),
                              '3-4_3-9': (((976.3, 1013.1), 'split_0'),),
                              '3-4_5-0': (((968.8, 1006.4), 'crossing_0'),),
                              '3-4_5-2': (((968.8, 1006.4), 'crossing_0'),),
                              '3-4_5-9': (((968.8, 1006.4), 'crossing_0'),),
                              '3-4_7-4': (((976.3, 1013), 'merging_0'),),
                              '3-6_3-9': (((976.7, 1010.8), 'split_0'),),
                              '3-6_5-0': (((977, 1001.9), 'crossing_0'),),
                              '3-6_5-2': (((977, 1001.9), 'crossing_0'),),
                              '3-6_5-6': (((974.1, 990.4), 'merging_0'),),
                              '3-6_5-9': (((977, 1001.9), 'crossing_0'),),
                              '3-6_7-4': (((976.3, 1013), 'crossing_0'),),
                              '3-9_5-0': (((982.4, 1001.4), 'merging_0'), ((993.7, 1004.3), 'split_0')),
                              '3-9_5-2': (((982.4, 1001.4), 'merging_0'), ((993.7, 1004.3), 'split_0')),
                              '3-9_5-9': (((982.4, 1001.4), 'merging_0'), ),
                              '3-9_7-0': (((993.7, 1004.3), 'crossing_0'),),
                              '3-9_7-2': (((993.7, 1004.3), 'crossing_0'),),
                              '3-9_7-4': (((993.7, 1004.3), 'crossing_0'), ((976.3, 1013), 'crossing_0'),),
                              '3-9_7-9': (((999, 1006.6), 'merging_0'),),
                              '3-9_8-9': (((1002.7, 1008.3), 'merging_0'),),
                              '5-0_5-2': (((984, 1025), 'split_0'),),
                              '5-0_5-6': (((972.8, 1003.5), 'split_0'),),
                              '5-0_5-9': (((993.7, 1004.4), 'split_0'),),
                              '5-0_7-0': (((993.7, 1004.4), 'merging_0'),),
                              '5-0_7-2': (((993.7, 1004.4), 'merging_0'), ((984, 1025.1), 'split_0')),
                              '5-0_7-4': (((993.7, 1004.4), 'merging_0'), ((990.4, 1019.8), 'split_0')),
                              '5-2_5-6': (((972.8, 1003.5), 'split_0'),),
                              '5-2_5-9': (((993.7, 1004.4), 'split_0'),),
                              '5-2_7-0': (((993.7, 1004.4), 'merging_0'), ((984, 1025.1), 'split_0')),
                              '5-2_7-2': (((993.7, 1004.4), 'merging_0'),),
                              '5-2_7-4': (((993.7, 1004.4), 'merging_0'), ((990.6, 1019.8), 'split_0')),
                              '5-6_5-9': (((972.8, 1003.5), 'split_0'),),
                              '5-9_7-0': (((993.7, 1004.3), 'crossing_0'),),
                              '5-9_7-2': (((993.7, 1004.3), 'crossing_0'),),
                              '5-9_7-4': (((968.8, 1006.4), 'crossing_0'), ((993.7, 1004.3), 'crossing_0')),
                              '5-9_7-9': (((999.4, 1006.8), 'merging_0'),),
                              '5-9_8-9': (((1002.7, 1008.3), 'merging_0'),),
                              '7-0_7-2': (((984, 1025.1), 'split_0'),),
                              '7-0_7-4': (((990.5, 1019.8), 'split_0'),),
                              '7-0_7-9': (((989.9, 1000.3), 'split_0'),),
                              '7-2_7-4': (((990.5, 1019.8), 'split_0'),),
                              '7-2_7-9': (((989.9, 1000.3), 'split_0'),),
                              '7-4_7-9': (((989.9, 1000.3), 'split_0'),),
                              '7-9_8-9': (((1002.7, 1008.3), 'merging_0'),),
                              }

    pickle_file = open('D:/Dev/UCB task/pickle/EPR/ref_path_xy_EPR.pkl', 'wb')
    pickle.dump(EPR_ref_path_points, pickle_file)
    pickle_file.close()

    ref_point_frenet = ref_paths2frenet(EPR_ref_path_points)
    pickle_file = open('D:/Dev/UCB task/pickle/EPR/ref_path_frenet_EPR.pkl', 'wb')
    pickle.dump(ref_point_frenet, pickle_file)
    pickle_file.close()

    EPR_interactions = dict()
    for k, vs in EPR_interaction_points.items():
        path1 = k.split('_')[0]
        path2 = k.split('_')[1]
        if path1 not in EPR_interactions.keys():
            EPR_interactions[path1] = dict()
        if path2 not in EPR_interactions.keys():
            EPR_interactions[path2] = dict()
        EPR_interactions[path1][path2] = []
        EPR_interactions[path2][path1] = []
        for v in vs:
            dis = cal_dis(np.array([list(v[0])]), np.array(EPR_ref_path_points[path1]))
            id1 = np.argmin(dis, axis=1)
            first_id = id1[0]
            dis2 = cal_dis(np.array([list(v[0])]), np.array(EPR_ref_path_points[path2]))
            id2 = np.argmin(dis2, axis=1)
            second_id = id2[0]
            # print(v[0], EPR_ref_path_points[path1][first_id], EPR_ref_path_points[path2][second_id])
            EPR_interactions[path1][path2].append((v[0], first_id, second_id, v[1]))
            EPR_interactions[path2][path1].append((v[0], second_id, first_id, v[1]))
    pickle_file = open('D:/Dev/UCB task/pickle/EPR/interaction_EPR.pkl', 'wb')
    pickle.dump(EPR_interactions, pickle_file)
    pickle_file.close()
    fig_dir = 'D:/Dev/UCB task/intersection_figs/'
    save_interaction_bg_figs(EPR_ref_path_points, EPR_interactions, map_path, fig_dir+'roundabout_EPR/')

    # generate intersection figures
    img_save_dir = fig_dir + 'roundabout_EPR_crop/'
    rotate_n = 0
    crop_interaction_figs(EPR_ref_path_points, EPR_interactions, ref_point_frenet, img_save_dir, rotate_n)
    xs = 950
    ys = 960
    save_complete_ref_path_fig(EPR_ref_path_points, fig_dir+'/single_EPR/', (xs, 1030), (ys, 1060))
    csv_data = get_track_label('D:/Dev/UCB task/Roundabout_EP_final/track_R/',
                               EPR_ref_path_points, ref_point_frenet)
    pickle_file = open('D:/Dev/UCB task/pickle/EPR/track_path_frenet_EPR.pkl', 'wb')
    pickle.dump(csv_data, pickle_file)
    pickle_file.close()
    save_ts_theta(csv_data, 'D:/Dev/UCB task/pickle/EPR/ts_theta_EPR.pkl')
    # save edge info
    for k, v in csv_data.items():
        print(k)
        split_edges = get_csv_edges(v, EPR_interactions, ref_point_frenet, k,
                                    img_save_dir + k + '/', EPR_ref_path_points)
        pickle_file = open('D:/Dev/UCB task/pickle/EPR/edges_EPR_{}.pkl'.format(k), 'wb')
        pickle.dump(split_edges, pickle_file)
        pickle_file.close()
