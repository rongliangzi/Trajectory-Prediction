from utils.EP_T_utils import *
from utils.roundabout_utils import fix_ref_path, save_interaction_bg_figs, save_complete_ref_path_fig,\
    crop_interaction_figs, ref_paths2frenet, save_ts_theta, get_csv_edges
import pickle


if __name__ == '__main__':
    func_file = 'D:/Dev/UCB task/Roundabout_EP_final/ref_path_funcs.txt'
    map_path = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/DR_USA_Roundabout_EP.osm'
    EPT_ref_path_points = read_funcs(func_file)
    insert_k = 2
    EPT_ref_path_points = fix_ref_path(EPT_ref_path_points, 'EPT', insert_k=insert_k)
    EPT_interaction_points = {'12-13_12-16': ((1054.42, 1008.51), 'split_0'),
                              '12-13_17-13': ((1082.95, 1006.94), 'merging_0'),
                              '12-16_14-11': ((1065.91, 1013.12), 'crossing_0'),
                              '12-16_15-16': ((1070.97, 1023.61), 'merging_0'),
                              '12-16_17-13': ((1068.58, 1016.41), 'crossing_0'),
                              '14-11_17-11': ((1054.20, 1013.76), 'merging_0'),
                              '14-11_17-13': ((1071.13, 1012.86), 'crossing_0'),
                              '17-11_17-13': ((1066.60, 1024.17), 'split_0')}
    EPT_interactions = dict()
    for k, v in EPT_interaction_points.items():
        path1 = k.split('_')[0]
        path2 = k.split('_')[1]
        if path1 not in EPT_interactions.keys():
            EPT_interactions[path1] = dict()
        if path2 not in EPT_interactions.keys():
            EPT_interactions[path2] = dict()
        dis = cal_dis(np.array([list(v[0])]), np.array(EPT_ref_path_points[path1]))
        id1 = np.argmin(dis, axis=1)
        first_id = id1[0]
        dis2 = cal_dis(np.array([list(v[0])]), np.array(EPT_ref_path_points[path2]))
        id2 = np.argmin(dis2, axis=1)
        second_id = id2[0]
        print(v[0], EPT_ref_path_points[path1][first_id], EPT_ref_path_points[path2][second_id])
        EPT_interactions[path1][path2] = [(v[0], first_id, second_id, v[1])]
        EPT_interactions[path2][path1] = [(v[0], second_id, first_id, v[1])]
    fig_dir = 'D:/Dev/UCB task/intersection_figs/'
    # save_interaction_bg_figs(EPT_ref_path_points, EPT_interactions, map_path, fig_dir+'roundabout_EPT/')
    ref_point_frenet = ref_paths2frenet(EPT_ref_path_points)
    # generate intersection figures
    img_save_dir = fig_dir + 'roundabout_EPT_crop/'
    rotate_n = 0
    # crop_interaction_figs(EPT_ref_path_points, EPT_interactions, ref_point_frenet, img_save_dir, rotate_n)
    xs = 1010
    ys = 990
    # save_complete_ref_path_fig(EPT_ref_path_points, fig_dir+'/single_EPT/', (xs, 1110), (ys, 1055))
    csv_data = get_track_label('D:/Dev/UCB task/Roundabout_EP_final/track_T/',
                               EPT_ref_path_points, ref_point_frenet)
    pickle_file = open('D:/Dev/UCB task/pickle/EPT/track_path_frenet_EPT.pkl', 'wb')
    pickle.dump(csv_data, pickle_file)
    pickle_file.close()
    save_ts_theta(csv_data, 'D:/Dev/UCB task/pickle/EPT/ts_theta_EPT.pkl')
    # save edge info
    for k, v in csv_data.items():
        print(k)
        split_edges = get_csv_edges(v, EPT_interactions, ref_point_frenet, k,
                                    img_save_dir + k + '/', EPT_ref_path_points)
        pickle_file = open('D:/Dev/UCB task/pickle/EPT/edges_EPT_{}.pkl'.format(k), 'wb')
        pickle.dump(split_edges, pickle_file)
        pickle_file.close()
