from utils.starting_area_utils import *
from utils import MA_utils
from utils.coordinate_transform import *
from utils.intersection_utils import find_intersection
from align_ref_img import counterclockwise_rotate
import pickle
import numpy as np
import csv


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
    ref_paths, csv_dict, rare_paths = MA_utils.get_ref_paths(data_base_path, data_dir_name)
    ref_path_info = dict()
    ref_path_info['ref_paths'] = ref_paths
    ref_path_info['csv_dict'] = csv_dict
    ref_path_info['rare_paths'] = rare_paths
    pickle_save_dir = save_base_dir + 'pickle/'
    pickle_file = open(pickle_save_dir + 'ref_path_info_new.pkl', 'wb')
    pickle.dump(ref_path_info, pickle_file)
    pickle_file.close()


def save_defined_ref_path_pickle():
    ref_paths, csv_dict = MA_utils.get_defined_ref_paths(defined_path_file, csv_path, x_start, y_start)
    ref_path_info = dict()
    ref_path_info['ref_paths'] = ref_paths
    ref_path_info['csv_dict'] = csv_dict
    ref_path_info['rare_paths'] = []
    pickle_save_dir = save_base_dir + 'pickle/'
    pickle_file = open(pickle_save_dir + 'ref_path_info_{}.pkl'.format(name), 'wb')
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


def gen_all_location_info(work_dir):
    ref_path_info_path = work_dir + 'pickle/ref_path_info_new.pkl'
    pickle_file = open(ref_path_info_path, 'rb')
    ref_path_info = pickle.load(pickle_file)
    pickle_file.close()
    ref_paths, csv_dict, rare_paths = ref_path_info['ref_paths'], ref_path_info['csv_dict'], ref_path_info['rare_paths']
    rare_paths += ['6-4']

    pickle_file = open(work_dir + 'pickle/ref_path_intersection.pkl', 'rb')
    intersection_info = pickle.load(pickle_file)
    pickle_file.close()

    pickle_file = open(work_dir + 'pickle/all_frenet_coordinate.pkl', 'rb')
    all_frenet = pickle.load(pickle_file)
    pickle_file.close()

    ref_frenet = dict()
    for path_name, path_points in ref_paths.items():
        start_path = int(path_name.split('-')[0])
        frenet_s_points, min_dis_id = ref_point2frenet(path_points[0], path_points[1],
                                                       new_coor_starting_area_dict[start_path]['stoplinex'],
                                                       new_coor_starting_area_dict[start_path]['stopliney'])
        ref_frenet[path_name] = dict()
        ref_frenet[path_name]['s'] = frenet_s_points
        ref_frenet[path_name]['min_dis_id'] = min_dis_id

    loc = dict()
    for csv_id, tracks in csv_dict.items():
        # for each csv, save a dict to pickle
        loc[csv_id] = dict()
        print(csv_id)
        for target, t_path in tracks:
            if t_path in rare_paths:
                continue
            t_id = target.track_id
            loc[csv_id][t_id] = dict()
            for ts in range(target.time_stamp_ms_first, target.time_stamp_ms_last + 100, 100):
                # save trajectory data
                loc[csv_id][t_id][ts] = dict()
                t_ms = target.motion_states[ts]
                theta = math.pi / 2 - t_ms.psi_rad
                target_s, target_d = all_frenet[csv_id][t_id][ts]
                # calculate the relative x,y and frenet coordinates of other cars
                for car, car_path in tracks:
                    # not for self
                    if car.track_id == t_id:
                        continue
                    # no intersection and not the same path, pass
                    if car_path not in intersection_info[t_path].keys() and car_path != t_path:
                        continue
                    # timestamp not included, pass
                    if ts not in car.motion_states.keys():
                        continue
                    car_s, car_d = all_frenet[csv_id][car.track_id][ts]
                    car_ms = car.motion_states[ts]
                    # counterclockwise rotate theta
                    car_x_rot, car_y_rot = counterclockwise_rotate(car_ms.x, car_ms.y, [t_ms.x, t_ms.y], theta)
                    new_coor = (car_x_rot - t_ms.x, car_y_rot - t_ms.y)
                    if car_path == t_path:
                        loc_info = (new_coor[0], new_coor[1], target_s, car_s, 1)
                    else:
                        intersection, first_id, second_id = intersection_info[t_path][car_path]
                        target_x_s = ref_frenet[t_path]['s'][first_id]
                        car_x_s = ref_frenet[car_path]['s'][second_id]
                        target_x_dis = target_x_s - target_s
                        car_x_dis = car_x_s - car_s
                        loc_info = (new_coor[0], new_coor[1], target_x_dis, car_x_dis, 0)
                    loc[csv_id][t_id][ts][car.track_id] = loc_info
                # finish one start timestamp
            # finish one target
        print('finish one csv!')
        pickle_save_dir = work_dir + 'pickle/'
        pickle_file = open(pickle_save_dir + 'all_location_info.pkl', 'wb')
        pickle.dump(loc, pickle_file)
        pickle_file.close()

    return


def gen_intersection_info(work_dir):
    ref_path_info_path = work_dir + 'pickle/ref_path_info_new.pkl'
    pickle_file = open(ref_path_info_path, 'rb')
    ref_path_info = pickle.load(pickle_file)
    pickle_file.close()
    ref_paths, rare_paths = ref_path_info['ref_paths'], ref_path_info['rare_paths']
    rare_paths += ['6-4']
    path_names = sorted(ref_paths.keys())
    intersection_info = dict()
    for path_name in path_names:
        intersection_info[path_name] = dict()
    for i in range(len(path_names)):
        path1 = path_names[i]
        if path1 in rare_paths:
            continue
        for j in range(i + 1, len(path_names)):
            path2 = path_names[j]
            if path2 in rare_paths:
                continue
            seq1 = ref_paths[path1]
            seq2 = ref_paths[path2]
            if path1.split('-')[0] == path2.split('-')[0]:
                continue
            intersection = find_intersection(seq1, seq2)
            if intersection is not None:
                # intersection of path1 and path2 exists
                intersection, first_id, second_id = intersection
                intersection_info[path1][path2] = (intersection, first_id, second_id)
                intersection_info[path2][path1] = (intersection, second_id, first_id)
    pickle_file = open(work_dir + 'pickle/ref_path_intersection.pkl', 'wb')
    pickle.dump(intersection_info, pickle_file)
    pickle_file.close()
    return


def gen_defined_intersection_info(work_dir):
    intersection_info = dict()
    with open(defined_intersection_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(list(csv_reader)[1:]):
            first_ref_path_id = int(row[1])
            second_ref_path_id = int(row[2])
            if first_ref_path_id not in intersection_info.keys():
                intersection_info[first_ref_path_id] = dict()
            if second_ref_path_id not in intersection_info.keys():
                intersection_info[second_ref_path_id] = dict()
            if row[3] == 'NULL':
                continue
            inter_x = float(row[3])
            inter_y = float(row[4])
            intersection_info[first_ref_path_id][second_ref_path_id] = ((inter_x, inter_y), None, None)
            intersection_info[second_ref_path_id][first_ref_path_id] = ((inter_x, inter_y), None, None)
    pickle_file = open(work_dir + 'pickle/intersection_{}.pkl'.format(name), 'wb')
    pickle.dump(intersection_info, pickle_file)
    pickle_file.close()


if __name__ == '__main__':
    data_base_path = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/'
    csv_path = 'D:/Dev/UCB task/track_updated/'
    # data_dir_name = 'DR_USA_Intersection_MA/'
    data_dir_name = 'DR_USA_Roundabout_EP'
    save_base_dir = 'D:/Dev/UCB task/'
    defined_path_file = 'D:/Dev/UCB task/RoundaboutEP_updated/ref_path_ROUNDABOUT_EP.csv'
    defined_intersection_file = 'D:/Dev/UCB task/RoundaboutEP_updated/interception_ROUNDABOUT_EP.csv'
    name = 'EP'
    x_start = 959
    y_start = 974
    save_defined_ref_path_pickle()
    gen_defined_intersection_info(save_base_dir)
    # save_ref_path_pickle()
    # plot_starting_area(save_base_dir)
    # gen_frenet(save_base_dir)
    # gen_all_location_info(save_base_dir)
    # gen_intersection_info(save_base_dir)
