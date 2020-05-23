from utils import map_vis_without_lanelet
from utils import ref_paths_utils
from utils.ref_paths_utils import starting_area_dict
from utils.intersection_utils import *
from utils import coordinate_transform
from utils.starting_area_utils import *
from align_ref_img import rotate_aug
import matplotlib.pyplot as plt
import os
import pickle
import math
import matplotlib.patches as patches


def plot_rotate_crop(ref_paths, path_name, path_info, fig_name, theta, start_point, save_dir):
    d = 4  # fig size
    r = 40  # range of x and y
    dpi = 50
    fig, axes = plt.subplots(1, 1, figsize=(d, d), dpi=dpi)
    # set bg to black
    axes.patch.set_facecolor("k")
    x1, y1, w1 = ref_paths[path_name][0], ref_paths[path_name][1], ref_paths[path_name][4]
    x1_rot, y1_rot = rotate_aug(x1, y1, start_point, theta)
    plt.plot(x1_rot, y1_rot, linewidth=w1 * 36 * d // r, color='r', zorder=25)
    for srd_path_name, intersection in path_info.items():
        x2, y2, w2 = ref_paths[srd_path_name][0], ref_paths[srd_path_name][1], ref_paths[srd_path_name][4]
        x2_rot, y2_rot = rotate_aug(x2, y2, start_point, theta)
        plt.plot(x2_rot, y2_rot, linewidth=w2 * 36 * d // r, color='b')
        intersection_x_rot, intersection_y_rot = rotate_aug(intersection[0], intersection[1], start_point, theta)
        circle = patches.Circle((intersection_x_rot, intersection_y_rot), (w1 + w2) * 2 * d / r, color='g', zorder=30)
        axes.add_patch(circle)
    # set x y range
    plt.xlim(start_point[0] - r//2, start_point[0] + r//2)
    plt.ylim(start_point[1], start_point[1] + r)

    # remove the white biankuang
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.savefig(save_dir+'{}.png'.format(fig_name))
    plt.close()


def plot_global_image(ref_paths, fig_name, work_dir, bg=True):
    '''
    :param ref_paths:
    :param fig_name: figure name
    :param work_dir: the dir to save
    :param bg: if plot background
    :return:
    '''
    d = 8  # fig size
    r = 120  # range of x and y
    dpi = 150
    xs, ys = x_s, y_s
    fig, axes = plt.subplots(1, 1, figsize=(d, d), dpi=dpi)
    if bg:
        # load and draw the lanelet2 map, either with or without the lanelet2 library
        lanelet_map_file = "D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/DR_USA_Intersection_MA.osm"
        map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, 0, 0)
    else:
        # set bg to black
        axes.patch.set_facecolor("k")
    for path_name in ref_paths.keys():
        x0, y0, _, _, w0 = ref_paths[path_name]
        x0 = x0 / 5 + x_s
        y0 = y0 / 5 + y_s
        plt.plot(x0, y0, linewidth=w0 * 36 * d // r, color='b')
    # set x y range
    plt.xlim(xs, xs + r)
    plt.ylim(ys, ys + r)

    # remove the white biankuang
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    save_dir = work_dir+'global_images/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.savefig(save_dir+'{}.png'.format(fig_name+'_bg' if bg else fig_name))
    plt.close()
    for path_name in ref_paths.keys():
        fig, axes = plt.subplots(1, 1, figsize=(d, d), dpi=dpi)
        lanelet_map_file = "D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/DR_USA_Intersection_MA.osm"
        map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, 0, 0)
        x0, y0, x1, y1, w0 = ref_paths[path_name]
        x0 = x0/5 + x_s
        y0 = y0/5 + y_s
        plt.plot(x0, y0, linewidth=w0 * 36 * d // r, color='b', alpha=0.5)
        plt.text(min(x0[0], 1084), y0[0], 'start', zorder=30)
        plt.text(min(x0[-1], 1084), y0[-1], 'end', zorder=30)
        x1 = x1 / 5 + x_s
        y1 = y1 / 5 + y_s
        plt.scatter(x1, y1, s=1)
        plt.xlim(xs, xs + r)
        plt.ylim(ys, ys + r)
        # remove the white biankuang
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(save_dir+'{}_{}.png'.format(fig_name, path_name))
        plt.close()


def main(work_dir):
    ref_path_info_path = work_dir + 'pickle/ref_path_info_new.pkl'
    pickle_file = open(ref_path_info_path, 'rb')
    ref_path_info = pickle.load(pickle_file)
    pickle_file.close()
    ref_paths, csv_dict, rare_paths = ref_path_info['ref_paths'], ref_path_info['csv_dict'], ref_path_info['rare_paths']
    # plot global image and each ref path
    plot_global_image(ref_paths, 'global', work_dir,)
    '''rare_paths += ['6-4']
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
            if intersection is None:
                continue
            else:
                # intersection of path1 and path2 exists
                intersection, first_id, second_id = intersection
                intersection_info[path1][path2] = intersection
                intersection_info[path2][path1] = intersection

    ref_path_id_dict = dict()
    for i, path_name in enumerate(path_names):
        ref_path_id_dict[path_name] = i
    pickle_save_dir = work_dir + 'pickle/'
    pickle_file = open(pickle_save_dir + 'ref_path_id_dict.pkl', 'wb')
    pickle.dump(ref_path_id_dict, pickle_file)
    pickle_file.close()
    for csv_id, tracks in csv_dict.items():
        # for each csv, save a dict to pickle
        coordinate_dict = dict()
        print(csv_id)
        for agent, path_name in tracks:
            if path_name in rare_paths:
                continue
            # for each agent(target car), save the agent to [ref_path_id][agent]
            ref_path_id = ref_path_id_dict[path_name]
            if ref_path_id not in coordinate_dict.keys():
                coordinate_dict[ref_path_id] = dict()
            coordinate_dict[ref_path_id][agent.track_id] = dict()
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
                # rotate the ref path around current point, crop (-20,20), (0,40), and save
                # plot_rotate_crop(ref_paths, path_name, intersection_info[path_name],
                #                  '{}_{}_{}_{}'.format(csv_id, str(agent.track_id), path_name, str(ts)),
                #                  theta, [ms.x, ms.y], cur_csv_work_dir)
                # save trajectory data
                coordinate_dict[ref_path_id][agent.track_id][start_ts] = dict()
                cur_path_points = None
                for ts in range(start_ts, start_ts+70*100, 100):
                    # in each frame
                    coordinate_dict[ref_path_id][agent.track_id][start_ts][ts] = dict()
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
                    coordinate_dict[ref_path_id][agent.track_id][start_ts][ts][agent.track_id] = (frenet_s, frenet_d)

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
                            coordinate_dict[ref_path_id][agent.track_id][start_ts][ts][car.track_id] = (frenet_s, frenet_d)

        pickle_save_dir = work_dir+'pickle/'
        pickle_file = open(pickle_save_dir+'{}_frenet_coordinate.pkl'.format(csv_id), 'wb')
        pickle.dump(coordinate_dict, pickle_file)
        pickle_file.close()
'''

    return


if __name__ == '__main__':
    data_base_path = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/'
    data_dir_name = 'DR_USA_Intersection_MA/'
    save_base_dir = 'D:/Dev/UCB task/'
    main(save_base_dir)
