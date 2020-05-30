from utils.coordinate_transform import get_frenet, get_xy
from gen_tools import ref_point2frenet
from utils.starting_area_utils import new_coor_starting_area_dict
from utils import dataset_reader
from utils import dict_utils
import pickle
import matplotlib.pyplot as plt


def visualize(raw_xy, trans_xy, old_xy):
    fig, axes = plt.subplots(1, 1, figsize=(20, 10), dpi=100)
    plt.subplot(1, 3, 1)
    plt.plot([xy[0] for xy in old_xy], [xy[1] for xy in old_xy])
    plt.title('Original coordinate')
    plt.subplot(1, 3, 2)
    raw_x = [xy[0] for xy in raw_xy]
    raw_y = [xy[1] for xy in raw_xy]
    plt.plot(raw_x, raw_y)
    plt.title('Pan and zoom coordinate')
    plt.subplot(1, 3, 3)
    plt.plot([xy[0] for xy in trans_xy], [xy[1] for xy in trans_xy])
    plt.title('Transformed from frenet')
    plt.show()


if __name__ == '__main__':
    work_dir = 'D:/Dev/UCB task/'
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

    tracks = csv_dict['000']
    trans_xy = []
    raw_xy = []
    trajectory_id = 60
    car, car_path = tracks[trajectory_id]
    for c, path in tracks:
        if c.track_id == trajectory_id:
            car, car_path = c, path
            break
    print(car.track_id)
    # print(car_path)
    ref_x, ref_y = ref_paths[car_path][0], ref_paths[car_path][1]
    ref_xy = list()
    for x, y in zip(ref_x, ref_y):
        ref_xy.append((x, y))
    for ts in range(car.time_stamp_ms_first, car.time_stamp_ms_last + 100, 100):
        x = car.motion_states[ts].x
        y = car.motion_states[ts].y
        psi_rad = car.motion_states[ts].psi_rad
        s, d = get_frenet(x, y, psi_rad, ref_xy, ref_frenet[car_path]['s'])
        trans_x, trans_y = get_xy(s, d, ref_frenet[car_path]['s'], ref_xy)
        raw_xy.append((x, y))
        trans_xy.append((trans_x, trans_y))

    track_dictionary = dataset_reader.read_tracks('D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/'+
                                                  'DR_USA_Intersection_MA/vehicle_tracks_000.csv')
    tracks = dict_utils.get_value_list(track_dictionary)
    agent = tracks[trajectory_id]
    for a in tracks:
        if a.track_id == trajectory_id:
            agent = a
    print(agent.track_id)
    old_xy = []
    for ts in range(agent.time_stamp_ms_first, agent.time_stamp_ms_last + 100, 100):
        old_xy.append((agent.motion_states[ts].x, agent.motion_states[ts].y))
    visualize(raw_xy, trans_xy, old_xy)
