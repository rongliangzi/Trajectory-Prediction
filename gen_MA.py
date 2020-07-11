from utils.MA_utils import *
from utils.roundabout_utils import plot_ref_path, ref_paths2frenet, \
    save_interaction_bg_figs, crop_interaction_figs, get_csv_edges, save_complete_ref_path_fig, save_ts_theta
import pickle
import json
import os
start_areas = dict()
start_areas[2] = dict()
start_areas[2]['x'] = [992, 992.2, 1002.3, 1002]
start_areas[2]['y'] = [1005.5, 1002.5, 1002, 1005.5]
start_areas[2]['stopline'] = [(1000.1, 1005.5), (1000.2, 1002.5)]
start_areas[3] = dict()
start_areas[3]['x'] = [992, 992.2, 1002.3, 1002]
start_areas[3]['y'] = [1002.5, 997.5, 993, 1002]
start_areas[3]['stopline'] = [(1000.15, 1001.5), (1000.3, 997)]
start_areas[6] = dict()
start_areas[6]['x'] = [1024.2, 1025.4, 1028.8, 1027.6]
start_areas[6]['y'] = [994.6, 983.8, 983.9, 995.1]
start_areas[6]['stopline'] = [(1024.7, 991.4), (1027.9, 991.6)]
start_areas[7] = dict()
start_areas[7]['x'] = [1028.3, 1029.1, 1036.6, 1037.7]
start_areas[7]['y'] = [994.8, 984.2, 984.4, 995.3]
start_areas[7]['stopline'] = [(1028.7, 991.6), (1036.7, 991.7)]
start_areas[9] = dict()
start_areas[9]['x'] = [1033.2, 1034.9, 1045.2, 1045.1]
start_areas[9]['y'] = [1015.7, 1005.9, 1006.5, 1010.1]
start_areas[9]['stopline'] = [(1037.8, 1006), (1036.6, 1012.7)]
start_areas[12] = dict()
start_areas[12]['x'] = [1015.3, 1015.5, 1020.5, 1019]
start_areas[12]['y'] = [1018, 1010.5, 1010.4, 1018.2]
start_areas[12]['stopline'] = [(1016.5, 1012), (1020, 1012)]
start_areas[13] = dict()
start_areas[13]['x'] = [1011.5, 1011.7, 1016, 1015.5]
start_areas[13]['y'] = [1018, 1010.5, 1010.4, 1018.2]
start_areas[13]['stopline'] = [(1011.7, 1012), (1016.3, 1012)]
start_areas[14] = dict()
start_areas[14]['x'] = [1006, 999, 1010.9, 1011.1]
start_areas[14]['y'] = [1020, 1011.3, 1011, 1020.2]
start_areas[14]['stopline'] = [(1004, 1012), (1011.3, 1012)]
end_areas = dict()
end_areas[1] = dict()
end_areas[1]['x'] = [988.3, 988.4, 999.9, 998.8]
end_areas[1]['y'] = [1009.9, 1006.3, 1005.7, 1012.2]
end_areas[4] = dict()
end_areas[4]['x'] = [1006.3, 1007.9, 1012.6, 1012.1]
end_areas[4]['y'] = [986.2, 975.5, 975.5, 985.8]
end_areas[5] = dict()
end_areas[5]['x'] = [1012.4, 1012.73, 1016, 1016]
end_areas[5]['y'] = [989.9, 976.9, 976.8, 990]
end_areas[8] = dict()
end_areas[8]['x'] = [1037.9, 1039.6, 1047.3, 1046.7]
end_areas[8]['y'] = [1005.5, 997.6, 999.1, 1006.3]
end_areas[10] = dict()
end_areas[10]['x'] = [1023.4, 1025.2, 1032.9, 1029.2]
end_areas[10]['y'] = [1025.7, 1014.9, 1015.7, 1025.8]
end_areas[11] = dict()
end_areas[11]['x'] = [1020.1, 1021.6, 1025.2, 1023.5]
end_areas[11]['y'] = [1023.3, 1012.3, 1012.6, 1024.5]
end_areas[15] = dict()
end_areas[15]['x'] = [1004.9, 1005.1, 1008, 1006.5]
end_areas[15]['y'] = [983.8, 975, 975.4, 983.4]
x_s, y_s = 971, 950

if __name__ == '__main__':
    map_dir = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/'
    map_name = "DR_USA_Intersection_MA.osm"
    base_path = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/'
    dir_name = 'DR_USA_Intersection_MA/'
    # plot_ref_path(map_dir+map_name, {}, start_areas, end_areas)

    if os.path.exists('D:/Dev/UCB task/pickle/MA/MA_ref_path_points.pkl') and \
            os.path.exists('D:/Dev/UCB task/pickle/MA/MA_csv_dict.pkl') and \
            os.path.exists('D:/Dev/UCB task/pickle/MA/rare_paths.pkl'):
        pickle_file = open('D:/Dev/UCB task/pickle/MA/MA_ref_path_points.pkl', 'rb')
        MA_ref_path_points = pickle.load(pickle_file)
        pickle_file.close()
        pickle_file = open('D:/Dev/UCB task/pickle/MA/MA_csv_dict.pkl', 'rb')
        csv_dict = pickle.load(pickle_file)
        pickle_file.close()
        pickle_file = open('D:/Dev/UCB task/pickle/MA/rare_paths.pkl', 'rb')
        rare_paths = pickle.load(pickle_file)
        pickle_file.close()
    else:
        # MA_ref_path_points[path name]: (x,2) array
        MA_ref_path_points, csv_dict, rare_paths = get_ref_paths(base_path, dir_name, start_areas,
                                                                 end_areas, x_s, y_s, save_img=True)
        pickle_file = open('D:/Dev/UCB task/pickle/MA/MA_ref_path_points.pkl', 'wb')
        pickle.dump(MA_ref_path_points, pickle_file)
        pickle_file.close()
        pickle_file = open('D:/Dev/UCB task/pickle/MA/MA_csv_dict.pkl', 'wb')
        pickle.dump(csv_dict, pickle_file)
        pickle_file.close()
        pickle_file = open('D:/Dev/UCB task/pickle/MA/rare_paths.pkl', 'wb')
        pickle.dump(rare_paths, pickle_file)
        pickle_file.close()
    ref_point_frenet = ref_paths2frenet(MA_ref_path_points)
    if os.path.exists('D:/Dev/UCB task/pickle/MA/track_path_frenet_MA.pkl'):
        pickle_file = open('D:/Dev/UCB task/pickle/MA/track_path_frenet_MA.pkl', 'rb')
        csv_data = pickle.load(pickle_file)
        pickle_file.close()
    else:
        csv_data = get_track_label(csv_dict, MA_ref_path_points, ref_point_frenet, rare_paths)
        pickle_file = open('D:/Dev/UCB task/pickle/MA/track_path_frenet_MA.pkl', 'wb')
        pickle.dump(csv_data, pickle_file)
        pickle_file.close()
    MA_interactions = find_ma_interactions(MA_ref_path_points, th=1, skip=30)
    # visualize the interactions with background
    # save_interaction_bg_figs(MA_ref_path_points, MA_interactions, map_dir + map_name,
    #                          'D:/Dev/UCB task/intersection_figs/roundabout_MA/')
    # generate interaction figures
    img_save_dir = 'D:/Dev/UCB task/intersection_figs/roundabout_MA_crop/'
    rotate_n = 0
    # crop_interaction_figs(MA_ref_path_points, MA_interactions, ref_point_frenet,
    #                       img_save_dir, rotate_n)
    save_complete_ref_path_fig(MA_ref_path_points, 'D:/Dev/UCB task/intersection_figs/single_MA/',
                               (955, 1105), (945, 1055))
    save_ts_theta(csv_data, 'D:/Dev/UCB task/pickle/MA/ts_theta_MA.pkl')
    '''# save edge info
    for k, v in csv_data.items():
        print(k)
        split_edges = get_csv_edges(v, MA_interactions, ref_point_frenet, k,
                                    img_save_dir+k+'/', MA_ref_path_points)
        if k in ['001', '002']:
            edge_keys = sorted(split_edges.keys())
            split_n = 4
            for i in range(split_n):
                end = min(len(edge_keys), (i+1)*len(edge_keys)//split_n)
                split_keys = edge_keys[i*len(edge_keys)//split_n:end]
                split_dict = dict()
                for key in split_keys:
                    split_dict[key] = split_edges[key]
                pickle_file = open('D:/Dev/UCB task/pickle/MA/edges_MA_{}_{}.pkl'.format(k, i), 'wb')
                pickle.dump(split_dict, pickle_file)
                pickle_file.close()
        else:
            pickle_file = open('D:/Dev/UCB task/pickle/MA/edges_MA_{}.pkl'.format(k), 'wb')
            pickle.dump(split_edges, pickle_file)
            pickle_file.close()'''
