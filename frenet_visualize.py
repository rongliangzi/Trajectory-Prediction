from utils.coordinate_transform import get_frenet, get_xy
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time


def visualize(raw_xy, trans_xy, ref_xy, delta_xy):
    fig, axes = plt.subplots(1, 1, figsize=(20, 10), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot([xy[0] for xy in raw_xy], [xy[1] for xy in raw_xy], 'r', marker='+')
    plt.plot([xy[0] for xy in ref_xy], [xy[1] for xy in ref_xy], 'g')
    plt.plot([xy[0] for xy in trans_xy], [xy[1] for xy in trans_xy], 'b', marker='+')
    plt.scatter(delta_xy[0][0], delta_xy[0][1], c='r')
    plt.scatter(delta_xy[1][0], delta_xy[1][1], c='b')
    plt.text(raw_xy[0][0], raw_xy[0][1], 'start', fontsize=20)
    plt.text(raw_xy[-1][0], raw_xy[-1][1], 'end', fontsize=20)
    plt.title('Original coordinate')
    plt.subplot(1, 2, 2)
    plt.plot([xy[0] for xy in trans_xy], [xy[1] for xy in trans_xy], marker='+')
    plt.plot([xy[0] for xy in ref_xy], [xy[1] for xy in ref_xy], 'g')
    plt.scatter(delta_xy[1][0], delta_xy[1][1], c='r')
    plt.title('Transformed from frenet')
    plt.show()
    plt.close()


def cal_max_delta(scene):
    ref_path_info_path = work_dir + 'pickle/{}/ref_path_xy_{}.pkl'.format(scene, scene)
    pickle_file = open(ref_path_info_path, 'rb')
    ref_paths = pickle.load(pickle_file)
    pickle_file.close()

    pickle_file = open(work_dir + 'pickle/{}/ref_path_frenet_{}.pkl'.format(scene, scene), 'rb')
    ref_frenet = pickle.load(pickle_file)
    pickle_file.close()

    pickle_file = open(work_dir + 'pickle/{}/track_path_frenet_{}.pkl'.format(scene, scene), 'rb')
    csv_data = pickle.load(pickle_file)
    pickle_file.close()
    delta_list = []
    start_t = time.time()
    for k, tracks in csv_data.items():
        print(k)
        # if k != '000':
        #     continue
        trans_xy = []
        raw_xy = []
        for trajectory_id, agent_dict in tracks.items():
            # if trajectory_id != 81:
            #     continue
            car_path = agent_dict['ref path']
            ref_xy = ref_paths[car_path]
            max_ts = 0
            max_delta = 0
            delta_xy = []
            for ts in range(agent_dict['time_stamp_ms_first'], agent_dict['time_stamp_ms_last'] + 100, 100):
            # for ts in range(256200-100, 256200 + 500, 100):
                x = agent_dict['motion_states'][ts]['x']
                y = agent_dict['motion_states'][ts]['y']
                psi_rad = agent_dict['motion_states'][ts]['psi_rad']
                # s = agent_dict['motion_states'][ts]['frenet_s']
                # d = agent_dict['motion_states'][ts]['frenet_d']
                s, d, _, _, _ = get_frenet(x, y, ref_xy, ref_frenet[car_path])
                trans_x, trans_y = get_xy(s, d, ref_frenet[car_path], ref_xy)
                raw_xy.append((x, y))
                trans_xy.append((trans_x, trans_y))
                delta = ((trans_x - x) ** 2 + (trans_y - y) ** 2) ** 0.5
                delta_list.append(delta)
                # if scene == 'FT' and car_path[0] != '9' and car_path not in ['11-6', '3-2', '7-6'] and max_delta < delta:
                if delta > max_delta:
                    max_delta = delta
                    max_ts = [ts, agent_dict['time_stamp_ms_first'], agent_dict['time_stamp_ms_last']]
                    delta_xy = [(x, y), (trans_x, trans_y)]
            if scene == 'FT' and max_delta > 0.01:# and car_path[0] != '9' and car_path not in ['11-6', '3-2', '7-6']:
                print(k, trajectory_id, max_ts, max_delta, car_path)
            elif scene == 'SR' and max_delta > 0.01:
                print(k, trajectory_id, max_ts, max_delta, car_path)
            elif scene == 'MA' and max_delta > 0.01:
                print(k, trajectory_id, max_ts, max_delta, car_path)
            # visualize(raw_xy, trans_xy, ref_xy, delta_xy)
    delta_list = np.array(delta_list)
    print('mean: ', delta_list.mean(), ', var: ', delta_list.var(), ', max: ', delta_list.max())
    print('time: ', time.time()-start_t)


if __name__ == '__main__':
    work_dir = 'D:/Dev/UCB task/'
    cal_max_delta('MA')
