import pickle
import math
from align_ref_img import counterclockwise_rotate
from utils.new_coor_ref_path_utils import *
from gen_tools import ref_point2frenet
import numpy as np
import time


def main(work_dir):
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
    start_time = time.time()
    for path_name, path_points in ref_paths.items():
        start_path = int(path_name.split('-')[0])
        frenet_s_points, min_dis_id = ref_point2frenet(path_points[0], path_points[1],
                                                       new_coor_starting_area_dict[start_path]['stoplinex'],
                                                       new_coor_starting_area_dict[start_path]['stopliney'])
        ref_frenet[path_name] = dict()
        ref_frenet[path_name]['s'] = frenet_s_points
        ref_frenet[path_name]['min_dis_id'] = min_dis_id
    print('time:', time.time()-start_time)
    loc = dict()
    for csv_id, tracks in csv_dict.items():
        # for each csv, save a dict to pickle
        loc[csv_id] = dict()
        print(csv_id)
        for target, t_path in tracks:
            if t_path in rare_paths:
                continue
            if t_path not in loc.keys():
                loc[csv_id][t_path] = dict()
            t_id = target.track_id
            loc[csv_id][t_path][t_id] = dict()
            start_id = int(t_path.split('-')[0])
            # if in starting area and have at least 69 frames behind, save ref path image and trajectory data
            for start_ts in range(target.time_stamp_ms_first, target.time_stamp_ms_last - 68*100, 100):
                ms = target.motion_states[start_ts]
                # judge if in starting area in starting frame
                polygon_points = new_coor_starting_area_dict[start_id]
                in_starting_area = judge_in_box(polygon_points['x'], polygon_points['y'], (ms.x, ms.y))
                # if not in starting area, pass
                if in_starting_area == 0:
                    continue
                theta = math.pi/2 - ms.psi_rad
                # save trajectory data
                loc[csv_id][t_path][t_id][start_ts] = dict()
                loc[csv_id][t_path][t_id][start_ts]['task'] = set()
                loc[csv_id][t_path][t_id][start_ts]['task'].add(t_path)
                # in 70 frames
                for ts in range(start_ts, start_ts + 70 * 100, 100):
                    loc[csv_id][t_path][t_id][start_ts][ts] = dict()
                    target_s, target_d = all_frenet[csv_id][t_id][ts]
                    # calculate the frenet coordinates of other cars and judge if in the box
                    for car, car_path in tracks:
                        if car.track_id == t_id or ts not in car.motion_states.keys() or \
                                car_path not in intersection_info[t_path].keys():
                            continue
                        # get the motion state of other car and judge if they are in the box
                        car_ms = car.motion_states[ts]
                        # counterclockwise rotate theta
                        car_x_rot, car_y_rot = counterclockwise_rotate(car_ms.x, car_ms.y, [ms.x, ms.y], theta)
                        new_coor = (car_x_rot - ms.x, car_y_rot - ms.y)
                        if -20 < new_coor[0] < 20 and 0 < new_coor[1] < 40:
                            car_s, car_d = all_frenet[csv_id][car.track_id][ts]
                            if car_s < -6:
                                continue
                            intersection, first_id, second_id = intersection_info[t_path][car_path]
                            target_x_s = ref_frenet[t_path]['s'][first_id]
                            car_x_s = ref_frenet[car_path]['s'][second_id]
                            target_x_dis = target_x_s - target_s
                            car_x_dis = car_x_s - car_s
                            loc[csv_id][t_path][t_id][start_ts][ts][car.track_id] = (new_coor[0], new_coor[1],
                                                                                     target_x_dis, car_x_dis)
                            loc[csv_id][t_path][t_id][start_ts]['task'].add(car_path)

        pickle_save_dir = work_dir+'pickle/'
        pickle_file = open(pickle_save_dir+'all_frenet_location.pkl', 'wb')
        pickle.dump(loc, pickle_file)
        pickle_file.close()

    return


if __name__ == '__main__':
    data_base_path = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/'
    data_dir_name = 'DR_USA_Intersection_MA/'
    save_base_dir = 'D:/Dev/UCB task/'
    main(save_base_dir)
