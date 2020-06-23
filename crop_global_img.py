from utils import map_vis_without_lanelet
from utils.intersection_utils import *
from utils import coordinate_transform
from utils.starting_area_utils import *
from utils.MA_utils import *
from align_ref_img import counterclockwise_rotate
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
    x1_rot, y1_rot = counterclockwise_rotate(x1, y1, start_point, theta)
    plt.plot(x1_rot, y1_rot, linewidth=w1 * 36 * d // r, color='r', zorder=25)
    for srd_path_name, intersection in path_info.items():
        x2, y2, w2 = ref_paths[srd_path_name][0], ref_paths[srd_path_name][1], ref_paths[srd_path_name][4]
        x2_rot, y2_rot = counterclockwise_rotate(x2, y2, start_point, theta)
        plt.plot(x2_rot, y2_rot, linewidth=w2 * 36 * d // r, color='b')
        intersection_x_rot, intersection_y_rot = counterclockwise_rotate(intersection[0], intersection[1], start_point, theta)
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
    rare_paths += ['6-4']

    # plot global image and each ref path
    plot_global_image(ref_paths, 'global', work_dir,)
    return


if __name__ == '__main__':
    data_base_path = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/'
    data_dir_name = 'DR_USA_Intersection_MA/'
    save_base_dir = 'D:/Dev/UCB task/'
    main(save_base_dir)
