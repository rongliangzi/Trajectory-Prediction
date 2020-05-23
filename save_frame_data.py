import pickle
import math
from align_ref_img import rotate_aug
from utils.ref_paths_utils import starting_area_dict
from utils import ref_paths_utils
from utils import coordinate_transform


def main(work_dir):
    ref_path_info_path = work_dir + 'pickle/ref_path_info.pkl'
    pickle_file = open(ref_path_info_path, 'rb')
    ref_path_info = pickle.load(pickle_file)
    pickle_file.close()
    ref_paths, csv_dict, rare_paths = ref_path_info['ref_paths'], ref_path_info['csv_dict'], ref_path_info['rare_paths']

    rare_paths += ['6-4']
    for csv_id, tracks in csv_dict.items():
        # for each csv, save a dict to pickle
        coordinate_dict = dict()
        print(csv_id)
        for agent, path_name in tracks:
            if path_name in rare_paths:
                continue
            if path_name not in coordinate_dict.keys():
                coordinate_dict[path_name] = dict()
            coordinate_dict[path_name][agent.track_id] = dict()
            start_id = int(path_name.split('-')[0])
            # if in starting area and have at least 69 frames behind, save ref path image and trajectory data
            for start_ts in range(agent.time_stamp_ms_first, agent.time_stamp_ms_last - 68*100, 100):
                ms = agent.motion_states[start_ts]
                # judge if in starting area in starting frame
                polygon_points = starting_area_dict[start_id]
                in_starting_area = ref_paths_utils.judge_in_box(polygon_points['x'], polygon_points['y'], (ms.x, ms.y))
                # if not in starting area
                if in_starting_area == 0:
                    continue
                # if in starting area, select as starting frame
                theta = math.pi/2 - ms.psi_rad
                # save trajectory data
                coordinate_dict[path_name][agent.track_id][start_ts] = dict()
                cur_path_points = None
                for ts in range(start_ts, start_ts+70*100, 100):
                    # in each frame
                    coordinate_dict[path_name][agent.track_id][start_ts][ts] = dict()
                    cur_ms = agent.motion_states[ts]
                    # save self frenet coordinate
                    if ts == start_ts:
                        x_points, y_points = ref_paths[path_name][0], ref_paths[path_name][1]
                        xy_points = list()
                        for x, y in zip(x_points, y_points):
                            xy_points.append((x, y))
                        # select the closest point as the starting ref path
                        path_point_start_id = coordinate_transform.closest_point_index(cur_ms.x, cur_ms.y, xy_points)
                        cur_path_points = xy_points[path_point_start_id:]
                    frenet_s, frenet_d = coordinate_transform.get_frenet(cur_ms.x, cur_ms.y, cur_ms.psi_rad,
                                                                         cur_path_points)
                    coordinate_dict[path_name][agent.track_id][start_ts][ts][agent.track_id] = (frenet_s, frenet_d)

                    # calculate the frenet coordinates of other cars and judge if in the box
                    for car, car_path_name in tracks:
                        if car.track_id == agent.track_id or ts not in car.motion_states.keys():
                            continue
                        # get the motion state of other car and judge if they are in the box
                        car_ms = car.motion_states[ts]
                        car_x_rot, car_y_rot = rotate_aug(car_ms.x, car_ms.y, [ms.x, ms.y], theta)
                        new_coor = (car_x_rot - ms.x, car_y_rot - ms.y)
                        if -20 < new_coor[0] < 20 and 0 < new_coor[1] < 40:
                            frenet_s, frenet_d = coordinate_transform.get_frenet(car_ms.x, car_ms.y, car_ms.psi_rad,
                                                                                 cur_path_points)
                            coordinate_dict[path_name][agent.track_id][start_ts][ts][car.track_id] = (frenet_s, frenet_d)

        pickle_save_dir = work_dir+'pickle/'
        pickle_file = open(pickle_save_dir+'{}_frenet_coordinate.pkl'.format(csv_id), 'wb')
        pickle.dump(coordinate_dict, pickle_file)
        pickle_file.close()

    return


if __name__ == '__main__':
    data_base_path = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/'
    data_dir_name = 'DR_USA_Intersection_MA/'
    save_base_dir = 'D:/Dev/UCB task/'
    main(save_base_dir)
