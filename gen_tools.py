from utils.starting_area_utils import *
from utils import new_coor_ref_path_utils
import pickle


def save_ref_path_pickle():
    ref_paths, csv_dict, rare_paths = new_coor_ref_path_utils.get_ref_paths(data_base_path, data_dir_name)
    ref_path_info = dict()
    ref_path_info['ref_paths'] = ref_paths
    ref_path_info['csv_dict'] = csv_dict
    ref_path_info['rare_paths'] = rare_paths
    pickle_save_dir = save_base_dir + 'pickle/'
    pickle_file = open(pickle_save_dir + 'ref_path_info_new.pkl', 'wb')
    pickle.dump(ref_path_info, pickle_file)
    pickle_file.close()


if __name__ == '__main__':
    data_base_path = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/'
    data_dir_name = 'DR_USA_Intersection_MA/'
    save_base_dir = 'D:/Dev/UCB task/'
    # save_ref_path_pickle()
    plot_starting_area(save_base_dir)

