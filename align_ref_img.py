from utils import ref_paths_utils
from utils.ref_paths_utils import starting_area_dict
from utils import coordinate_transform
from utils import map_vis_without_lanelet
from utils.intersection_utils import *
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import pickle


def rotate_aug(x, y, intersection, theta):
    # rotate the x,y point counterclockwise at angle theta around intersection point
    x_rot = (x - intersection[0]) * math.cos(theta) - (y - intersection[1]) * math.sin(theta) + intersection[0]
    y_rot = (x - intersection[0]) * math.sin(theta) + (y - intersection[1]) * math.cos(theta) + intersection[1]
    return x_rot, y_rot


def plot_intersection(x1, y1, w1, x2, y2, w2, fig_name, intersection, id1, id2, work_dir, bg=True):
    '''
    :param x1: x list
    :param y1: y list
    :param w1: line width
    :param x2:
    :param y2:
    :param w2:
    :param fig_name: figure name to save
    :param intersection: (x,y) of intersection
    :param id1: id of the intersection point in x1
    :param id2:
    :param work_dir: directory to save figure
    :return: save the figure
    '''
    d = 8  # fig size
    r = 80  # range of x and y
    al = 8  # arrow length
    fig, axes = plt.subplots(1, 1, figsize=(d, d), dpi=100)
    if bg:
        lanelet_map_file = "D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/DR_USA_Intersection_MA.osm"
        map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, 0, 0)
    else:
        # set bg to black
        axes.patch.set_facecolor("k")
    circle = patches.Circle(intersection, (w1 + w2) * 2 * d / r, color='r', zorder=3)
    axes.add_patch(circle)
    # calculate the k as the tangent
    if id1+1 >= len(x1):
        delta_y1, delta_x1 = y1[id1] - y1[id1 - 1], x1[id1] - x1[id1 - 1]
    elif id1-1 < 0:
        delta_y1, delta_x1 = y1[id1 + 1] - y1[id1], x1[id1 + 1] - x1[id1]
    else:
        delta_y1, delta_x1 = y1[id1 + 1] - y1[id1 - 1], x1[id1 + 1] - x1[id1 - 1]
    theta1 = math.atan2(delta_y1, delta_x1)
    # convert from -pi~pi to 0~2pi
    if theta1 < 0:
        theta1 += 2*math.pi
    if id2+1 >= len(x2):
        delta_y2, delta_x2 = (y2[id2] - y2[id2 - 1]), (x2[id2] - x2[id2 - 1])
    elif id2-1 < 0:
        delta_y2, delta_x2 = (y2[id2 + 1] - y2[id2]), (x2[id2 + 1] - x2[id2])
    else:
        delta_y2, delta_x2 = (y2[id2 + 1] - y2[id2 - 1]), (x2[id2 + 1] - x2[id2 - 1])
    theta2 = math.atan2(delta_y2, delta_x2)
    # convert from -pi~pi to 0~2pi
    if theta2 < 0:
        theta2 += 2*math.pi
    # before rotation
    plt.plot(x1, y1, linewidth=w1 * 72 * d // r, color='b')
    plt.plot(x2, y2, linewidth=w2 * 72 * d // r, color='g')
    delta_xy1 = (delta_x1 ** 2 + delta_y1 ** 2) ** 0.5
    ar_x1, ar_y1 = delta_x1 / delta_xy1, delta_y1 / delta_xy1
    axes.arrow(x1[id1], y1[id1], al * ar_x1, al * ar_y1, zorder=30, color='purple', width=0.2, head_width=0.6)
    delta_xy2 = (delta_x2 ** 2 + delta_y2 ** 2) ** 0.5
    ar_x2, ar_y2 = delta_x2 / delta_xy2, delta_y2 / delta_xy2
    axes.arrow(x2[id2], y2[id2], al * ar_x2, al * ar_y2, zorder=30, color='yellow', width=0.2, head_width=0.6)

    # theta of angle bisector
    avg_theta = (theta1+theta2)/2
    theta1_rot = theta1 - avg_theta
    k1_rot = math.tan(theta1_rot)
    theta2_rot = theta2 - avg_theta
    k2_rot = math.tan(theta2_rot)

    # draw the arrow whose length is al
    # rotate according to angle bisector to align
    x1_rot, y1_rot = rotate_aug(x1, y1, intersection, -avg_theta)
    x2_rot, y2_rot = rotate_aug(x2, y2, intersection, -avg_theta)
    plt.plot(x1_rot, y1_rot, linewidth=w1 * 72 * d // r, color='b')
    plt.plot(x2_rot, y2_rot, linewidth=w2 * 72 * d // r, color='g')
    axes.arrow(x1[id1], y1[id1], al/(k1_rot**2+1)**0.5, al * k1_rot/(k1_rot**2+1)**0.5, zorder=4,
               color='purple', width=0.2, head_width=0.6)
    axes.arrow(x2[id2], y2[id2], al/(k2_rot**2+1)**0.5, al * k2_rot/(k2_rot**2+1)**0.5, zorder=5,
               color='yellow', width=0.2, head_width=0.6)

    # set x y range
    plt.xlim(intersection[0]-r//2, intersection[0]+r//2)
    plt.ylim(intersection[1]-r//2, intersection[1]+r//2)

    # remove the white biankuang
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    save_dir = work_dir + 'intersection_figs/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if bg:
        fig_name = fig_name + '_bg'
    plt.savefig(save_dir+'{}.png'.format(fig_name))
    plt.close()


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


def save_pickle(path_names, work_dir, csv_dict, rare_paths, ref_paths):
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
            for start_ts in range(agent.time_stamp_ms_first, agent.time_stamp_ms_last - 69 * 100, 100):
                ms = agent.motion_states[start_ts]
                # judge if in starting area in starting frame
                polygon_points = starting_area_dict[start_id]
                in_starting_area = ref_paths_utils.judge_in_box(polygon_points['x'], polygon_points['y'], (ms.x, ms.y))
                # if not in starting area
                if in_starting_area == 0:
                    continue
                # if in starting area, select as starting frame
                theta = math.pi / 2 - ms.psi_rad
                # save trajectory data
                coordinate_dict[ref_path_id][agent.track_id][start_ts] = dict()
                cur_path_points = None
                for ts in range(start_ts, start_ts + 70 * 100, 100):
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
                            coordinate_dict[ref_path_id][agent.track_id][start_ts][ts][car.track_id] = (
                            frenet_s, frenet_d)

        pickle_save_dir = work_dir + 'pickle/'
        pickle_file = open(pickle_save_dir + '{}_frenet_coordinate.pkl'.format(csv_id), 'wb')
        pickle.dump(coordinate_dict, pickle_file)
        pickle_file.close()


def main(work_dir):
    ref_path_info_path = work_dir + 'pickle/ref_path_info.pkl'
    pickle_file = open(ref_path_info_path, 'rb')
    ref_path_info = pickle.load(pickle_file)
    pickle_file.close()
    ref_paths, csv_dict, rare_paths = ref_path_info['ref_paths'], ref_path_info['csv_dict'], ref_path_info['rare_paths']
    rare_paths += ['6-4']
    path_names = sorted(ref_paths.keys())
    intersection_info = dict()
    for path_name in path_names:
        intersection_info[path_name] = dict()
    img_count = 0
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
                # no intersection
                continue
            else:
                # intersection of path1 and path2 exists
                intersection, first_id, second_id = intersection
                intersection_info[path1][path2] = intersection
                intersection_info[path2][path1] = intersection

                plot_intersection(seq1[0], seq1[1], seq1[4], seq2[0], seq2[1], seq2[4], path1+' '+path2,
                                  intersection, first_id, second_id, work_dir)
                degree = 2
                # rotate x1,y1 +- degree around intersection and plot
                theta = math.pi*degree/180
                x1_rot, y1_rot = rotate_aug(seq1[0], seq1[1], intersection, theta)
                plot_intersection(x1_rot, y1_rot, seq1[4], seq2[0], seq2[1], seq2[4], path1+' '+path2+'_'+str(degree),
                                  intersection, first_id, second_id, work_dir)

                x1_rot, y1_rot = rotate_aug(seq1[0], seq1[1], intersection, -theta)
                plot_intersection(x1_rot, y1_rot, seq1[4], seq2[0], seq2[1], seq2[4],
                                  path1 + ' ' + path2 + '_-' + str(degree),
                                  intersection, first_id, second_id, work_dir)
                img_count += 3
                print(img_count)
                # break
        # break
    return


def save_ref_path_pickle():
    ref_paths, csv_dict, rare_paths = ref_paths_utils.get_ref_paths(data_base_path, data_dir_name)
    ref_path_info = dict()
    ref_path_info['ref_paths'] = ref_paths
    ref_path_info['csv_dict'] = csv_dict
    ref_path_info['rare_paths'] = rare_paths
    pickle_save_dir = save_base_dir + 'pickle/'
    pickle_file = open(pickle_save_dir + 'ref_path_info.pkl', 'wb')
    pickle.dump(ref_path_info, pickle_file)
    pickle_file.close()


if __name__ == '__main__':
    data_base_path = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/'
    data_dir_name = 'DR_USA_Intersection_MA/'
    save_base_dir = 'D:/Dev/UCB task/'
    # save_ref_path_pickle()
    main(save_base_dir)
