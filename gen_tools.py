from utils.starting_area_utils import *
from utils import new_coor_ref_path_utils
from utils.coordinate_transform import *
import pickle
import numpy as np


# transform points in a ref path to frenet coordinate
def ref_point2frenet(x_points, y_points, stoplinex, stopliney):
    k = (stopliney[1] - stopliney[0]) / (stoplinex[1] - stoplinex[0])
    b = (stoplinex[1] * stopliney[0] - stoplinex[0] * stopliney[1]) / (stoplinex[1] - stoplinex[0])
    min_dis = 1e8
    min_dis_id = -1
    for i in range(len(x_points)):
        dis = abs(k*x_points[i]-y_points[i]+b)
        if dis < min_dis:
            min_dis = dis
            min_dis_id = i
    frenet_s_points = [0]
    for i in range(1, len(x_points)):
        prev_x = x_points[i-1]
        prev_y = y_points[i-1]
        x = x_points[i]
        y = y_points[i]
        s_dis = ((x-prev_x)**2+(y-prev_y)**2)**0.5
        frenet_s_points.append(frenet_s_points[i-1]+s_dis)
    delta = frenet_s_points[min_dis_id]
    frenet_s_points = [s-delta for s in frenet_s_points]
    frenet_s_points = np.array(frenet_s_points)
    return frenet_s_points, min_dis_id


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


def gen_frenet(work_dir):
    ref_path_info_path = work_dir + 'pickle/ref_path_info_new.pkl'
    pickle_file = open(ref_path_info_path, 'rb')
    ref_path_info = pickle.load(pickle_file)
    pickle_file.close()
    ref_paths, csv_dict, rare_paths = ref_path_info['ref_paths'], ref_path_info['csv_dict'], ref_path_info['rare_paths']
    rare_paths += ['6-4']

    ref_frenet = dict()
    for path_name, path_points in ref_paths.items():
        start_path = int(path_name.split('-')[0])
        frenet_s_points, min_dis_id = ref_point2frenet(path_points[0], path_points[1],
                                                       new_coor_starting_area_dict[start_path]['stoplinex'],
                                                       new_coor_starting_area_dict[start_path]['stopliney'])
        ref_frenet[path_name] = dict()
        ref_frenet[path_name]['s'] = frenet_s_points
        ref_frenet[path_name]['min_dis_id'] = min_dis_id

    # for each csv
    frenet = dict()
    for csv_id, tracks in csv_dict.items():
        print(csv_id)
        frenet[csv_id] = dict()
        for car, car_path in tracks:
            # print(car_path)
            frenet[csv_id][car.track_id] = dict()
            x_points, y_points = ref_paths[car_path][0], ref_paths[car_path][1]
            xy_points = list()
            for x, y in zip(x_points, y_points):
                xy_points.append((x, y))
            for ts in range(car.time_stamp_ms_first, car.time_stamp_ms_last+100, 100):
                x = car.motion_states[ts].x
                y = car.motion_states[ts].y
                psi_rad = car.motion_states[ts].psi_rad
                frenet[csv_id][car.track_id][ts] = get_frenet(x, y, psi_rad, xy_points, ref_frenet[car_path]['s'])
    pickle_save_dir = work_dir + 'pickle/'
    pickle_file = open(pickle_save_dir + 'all_frenet_coordinate.pkl', 'wb')
    pickle.dump(frenet, pickle_file)
    pickle_file.close()


if __name__ == '__main__':
    data_base_path = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/'
    data_dir_name = 'DR_USA_Intersection_MA/'
    save_base_dir = 'D:/Dev/UCB task/'
    # save_ref_path_pickle()
    # plot_starting_area(save_base_dir)
    gen_frenet(save_base_dir)
