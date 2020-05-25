from utils import new_coor_ref_path_utils
from utils import coordinate_transform
from utils import map_vis_without_lanelet
from utils.starting_area_utils import *
from utils.intersection_utils import *
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import pickle


def counterclockwise_rotate(x, y, intersection, theta):
    # rotate the x,y point counterclockwise at angle theta around intersection point
    x_rot = (x - intersection[0]) * math.cos(theta) - (y - intersection[1]) * math.sin(theta) + intersection[0]
    y_rot = (x - intersection[0]) * math.sin(theta) + (y - intersection[1]) * math.cos(theta) + intersection[1]
    return x_rot, y_rot


def plot_aligned_img(x1, y1, w1, x2, y2, w2, fig_name, intersection, id1, id2, work_dir, bg=False):
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
    :param bg: if plot background
    :return: save the figure
    '''
    d = 8  # fig size
    r = 200  # range of x and y
    al = 8  # arrow length
    dpi = 25
    fig, axes = plt.subplots(1, 1, figsize=(d, d), dpi=dpi)
    if bg:
        lanelet_map_file = "D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/DR_USA_Intersection_MA.osm"
        map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, 0, 0)
    else:
        # set bg to black
        axes.patch.set_facecolor("k")
    circle = patches.Circle(intersection, (w1 + w2) * 6 * rate * d / r, color='r', zorder=3)
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
    if bg:
        # before rotation
        plt.plot(x1, y1, linewidth=w1 * 72 * rate * d // r, color='b')
        plt.plot(x2, y2, linewidth=w2 * 72 * rate * d // r, color='g')
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
    x1_rot, y1_rot = counterclockwise_rotate(x1, y1, intersection, -avg_theta)
    x2_rot, y2_rot = counterclockwise_rotate(x2, y2, intersection, -avg_theta)
    plt.plot(x1_rot, y1_rot, linewidth=w1 * 72 * rate * d // r, color='b')
    plt.plot(x2_rot, y2_rot, linewidth=w2 * 72 * rate * d // r, color='g')
    if bg:
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
    return avg_theta


def main(work_dir):
    ref_path_info_path = work_dir + 'pickle/ref_path_info_new.pkl'
    pickle_file = open(ref_path_info_path, 'rb')
    ref_path_info = pickle.load(pickle_file)
    pickle_file.close()
    ref_paths, csv_dict, rare_paths = ref_path_info['ref_paths'], ref_path_info['csv_dict'], ref_path_info['rare_paths']
    rare_paths += ['6-4']
    path_names = sorted(ref_paths.keys())
    intersection_info = dict()
    for path_name in path_names:
        intersection_info[path_name] = dict()
    ref_path_img_theta = dict()
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

            if intersection is not None:
                # intersection of path1 and path2 exists
                intersection, first_id, second_id = intersection
                intersection_info[path1][path2] = (intersection, first_id, second_id)
                intersection_info[path2][path1] = (intersection, second_id, first_id)

                theta = plot_aligned_img(seq1[0], seq1[1], seq1[4], seq2[0], seq2[1], seq2[4], path1+' '+path2,
                                         intersection, first_id, second_id, work_dir)
                ref_path_img_theta[path1+'_'+path2] = theta
                degree = 2
                # rotate x1,y1 +- degree around intersection and plot
                theta = math.pi*degree/180
                x1_rot, y1_rot = counterclockwise_rotate(seq1[0], seq1[1], intersection, theta)
                plot_aligned_img(x1_rot, y1_rot, seq1[4], seq2[0], seq2[1], seq2[4],
                                          path1+' '+path2+'_'+str(degree), intersection, first_id, second_id, work_dir)

                x1_rot, y1_rot = counterclockwise_rotate(seq1[0], seq1[1], intersection, -theta)
                plot_aligned_img(x1_rot, y1_rot, seq1[4], seq2[0], seq2[1], seq2[4],
                                  path1 + ' ' + path2 + '_-' + str(degree),
                                  intersection, first_id, second_id, work_dir)
                img_count += 3
                print(img_count)
                # break
        # break
    pickle_save_dir = work_dir + 'pickle/'
    pickle_file = open(pickle_save_dir + 'ref_path_img_theta.pkl', 'wb')
    pickle.dump(ref_path_img_theta, pickle_file)
    pickle_file.close()
    pickle_file = open(pickle_save_dir + 'ref_path_intersection.pkl', 'wb')
    pickle.dump(intersection_info, pickle_file)
    pickle_file.close()
    return


if __name__ == '__main__':
    data_base_path = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/'
    data_dir_name = 'DR_USA_Intersection_MA/'
    save_base_dir = 'D:/Dev/UCB task/'
    main(save_base_dir)
