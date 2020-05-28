import pickle
import math
from align_ref_img import counterclockwise_rotate
from utils.new_coor_ref_path_utils import *
from gen_tools import ref_point2frenet
from utils.starting_area_utils import rate
import numpy as np
import os
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

    pickle_file = open(work_dir + 'pickle/all_location_info.pkl', 'rb')
    all_loc = pickle.load(pickle_file)
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

    for csv_id, tracks in csv_dict.items():
        # for each csv, save a dict to pickle
        loc = dict()
        start_time = time.time()
        print(csv_id)
        for target, t_path in tracks:
            if t_path in rare_paths:
                continue
            t_id = target.track_id
            loc[t_id] = dict()
            loc[t_id]['ref path'] = t_path
            start_id = int(t_path.split('-')[0])
            # if in starting area and have at least 69 frames behind, save ref path image and trajectory data
            for start_ts in range(target.time_stamp_ms_first, target.time_stamp_ms_last - 68 * 100, 100):
                start_ms = target.motion_states[start_ts]
                # judge if in starting area in starting frame
                polygon_points = new_coor_starting_area_dict[start_id]
                in_starting_area = judge_in_box(polygon_points['x'], polygon_points['y'], (start_ms.x, start_ms.y))
                # if not in starting area, pass
                if in_starting_area == 0:
                    continue
                # save trajectory data
                loc[t_id][start_ts] = dict()
                loc[t_id][start_ts]['task'] = set()
                loc[t_id][start_ts]['task'].add(t_path)
                loc[t_id][start_ts]['agent'] = [(t_id, t_path)]
                ts_20 = start_ts + 19 * 100
                t_20_ms = target.motion_states[ts_20]
                t_20_s, _ = all_frenet[csv_id][t_id][ts_20]
                # find surrounding cars in the 20th frame, save ref path in 'task' and track id, type in 'agent'
                for car, car_path in tracks:
                    # not self, containing this timestamp
                    if car.track_id == t_id or ts_20 not in car.motion_states.keys():
                        continue
                    car_20_s, car_20_d = all_frenet[csv_id][car.track_id][ts_20]
                    # far from stopline
                    if car_20_s < -6 * rate:
                        continue
                    # no intersection with any other ref path
                    if len(intersection_info[car_path].keys()) == 0:
                        continue
                    # behind target and no intersection with target
                    # if (car_path == t_path and car_20_s < t_20_s) or car_path not in intersection_info[t_path].keys():
                    #     continue
                    # get the motion state of other car and judge if they are in the box
                    car_ms = car.motion_states[ts_20]
                    # counterclockwise rotate theta
                    theta = math.pi / 2 - t_20_ms.psi_rad
                    car_x_rot, car_y_rot = counterclockwise_rotate(car_ms.x, car_ms.y, [t_20_ms.x, t_20_ms.y], theta)
                    new_coor = (car_x_rot - t_20_ms.x, car_y_rot - t_20_ms.y)
                    if -20 * rate < new_coor[0] < 20 * rate and 0 < new_coor[1] < 40 * rate:
                        loc[t_id][start_ts]['task'].add(car_path)
                        loc[t_id][start_ts]['agent'].append((car.track_id, car_path))

                # delete the ref paths that have no intersection with any other ref path in this case
                case_path_names = sorted(loc[t_id][start_ts]['task'])
                img_names = set()
                to_delete_path = []
                for path1 in case_path_names:
                    intersection_num = 0
                    for path2 in case_path_names:
                        if path2 == path1:
                            continue
                        if path2 in intersection_info[path1].keys():
                            intersection_num += 1
                            if path1 < path2:
                                img_names.add(path1 + ' ' + path2)
                    if intersection_num == 0:
                        to_delete_path.append(path1)
                # save image name set in 'task'
                loc[t_id][start_ts]['task'] = img_names
                # delete all the agents whose ref paths have no intersection with other agents.
                for ag_i in range(len(loc[t_id][start_ts]['agent']) - 1, -1, -1):
                    ag_id, ag_path = loc[t_id][start_ts]['agent'][ag_i]
                    if ag_path in to_delete_path:
                        del loc[t_id][start_ts]['agent'][ag_i]
                agent_list = list()
                for ag_id, ag_path in loc[t_id][start_ts]['agent']:
                    agent_list.append(ag_id)
                # no agents with intersection, omit this case
                if len(agent_list) < 2:
                    del loc[t_id][start_ts]
                    continue

                # save 70 trajectory for all agents
                loc[t_id][start_ts]['trajectory'] = dict()
                for agent_id, agent_path in loc[t_id][start_ts]['agent']:
                    trajectory_list = []
                    for ts in range(start_ts, start_ts + 70 * 100, 100):
                        if ts not in all_frenet[csv_id][agent_id].keys():
                            trajectory_list.append('NaN')
                            # print(agent_path, start_ts, ts)
                        else:
                            trajectory_list.append(all_frenet[csv_id][agent_id][ts])
                    loc[t_id][start_ts]['trajectory'][agent_id] = trajectory_list
                # in 70 frames
                for ts in range(start_ts, start_ts + 70 * 100, 100):
                    loc[t_id][start_ts][ts] = dict()
                    # calculate the frenet coordinates of other cars and judge if in the box
                    for car1_id in agent_list:
                        for car2_id in agent_list:
                            if car1_id == car2_id:
                                continue
                            if ts not in all_loc[csv_id][car1_id].keys():
                                continue
                            if car2_id not in all_loc[csv_id][car1_id][ts].keys():
                                continue
                            new_x, new_y, s, d, tp = all_loc[csv_id][car1_id][ts][car2_id]
                            loc_info = (new_x, new_y, s, d, tp)
                            if car1_id not in loc[t_id][start_ts][ts].keys():
                                loc[t_id][start_ts][ts][car1_id] = dict()
                            loc[t_id][start_ts][ts][car1_id][car2_id] = loc_info
                # finish one start timestamp
            # finish one target
        pickle_save_dir = work_dir+'pickle/relative_coordinate_s_distance/'
        if not os.path.exists(pickle_save_dir):
            os.mkdir(pickle_save_dir)
        pickle_file = open(pickle_save_dir+'{}_relative_coor_s_dis.pkl'.format(csv_id), 'wb')
        pickle.dump(loc, pickle_file)
        pickle_file.close()
        print('finish one csv! time:', time.time() - start_time)

    return


if __name__ == '__main__':
    data_base_path = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/'
    data_dir_name = 'DR_USA_Intersection_MA/'
    save_base_dir = 'D:/Dev/UCB task/'
    main(save_base_dir)
