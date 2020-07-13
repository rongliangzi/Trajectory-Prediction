from utils.coordinate_transform import get_frenet, get_xy
import pickle
import matplotlib.pyplot as plt


def visualize(raw_xy, trans_xy, ref_xy):
    fig, axes = plt.subplots(1, 1, figsize=(20, 10), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot([xy[0] for xy in raw_xy], [xy[1] for xy in raw_xy], marker='+')
    plt.plot([xy[0] for xy in ref_xy], [xy[1] for xy in ref_xy], 'g')
    plt.text(raw_xy[0][0], raw_xy[0][1], 'start', fontsize=20)
    plt.text(raw_xy[-1][0], raw_xy[-1][1], 'end', fontsize=20)
    plt.title('Original coordinate')
    plt.subplot(1, 2, 2)
    plt.plot([xy[0] for xy in trans_xy], [xy[1] for xy in trans_xy], marker='+')
    plt.plot([xy[0] for xy in ref_xy], [xy[1] for xy in ref_xy], 'g')
    plt.title('Transformed from frenet')
    plt.show()


if __name__ == '__main__':
    work_dir = 'D:/Dev/UCB task/'
    ref_path_info_path = work_dir + 'pickle/SR/ref_path_xy_SR.pkl'
    pickle_file = open(ref_path_info_path, 'rb')
    ref_paths = pickle.load(pickle_file)
    pickle_file.close()

    pickle_file = open(work_dir + 'pickle/SR/ref_path_frenet_SR.pkl', 'rb')
    ref_frenet = pickle.load(pickle_file)
    pickle_file.close()

    pickle_file = open(work_dir + 'pickle/SR/track_path_frenet_SR.pkl', 'rb')
    csv_data = pickle.load(pickle_file)
    pickle_file.close()

    tracks = csv_data['000']
    trans_xy = []
    raw_xy = []
    trajectory_id = 50
    agent_dict = tracks[trajectory_id]
    car_path = agent_dict['ref path']
    # print(car_path)
    ref_xy = ref_paths[car_path]
    max_delta = 0
    for ts in range(agent_dict['time_stamp_ms_first'], agent_dict['time_stamp_ms_last']+100, 100):
        x = agent_dict['motion_states'][ts]['x']
        y = agent_dict['motion_states'][ts]['y']
        psi_rad = agent_dict['motion_states'][ts]['psi_rad']
        # s = agent_dict['motion_states'][ts]['frenet_s']
        # d = agent_dict['motion_states'][ts]['frenet_d']
        s, d, proj = get_frenet(x, y, psi_rad, ref_xy, ref_frenet[car_path])
        trans_x, trans_y = get_xy(s, d, ref_frenet[car_path], ref_xy, proj)
        max_delta = max(max_delta, ((trans_x-x)**2+(trans_y-y)**2)**0.5)
        raw_xy.append((x, y))
        trans_xy.append((trans_x, trans_y))
    print(max_delta)
    visualize(raw_xy, trans_xy, ref_xy)
