from utils import map_vis_without_lanelet
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.io as scio
import numpy as np
import math


def plot_ref_path(map_file, all_points, circle_point):
    fig, axes = plt.subplots(1, 1)
    map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
    for way_points in all_points[0, :]:
        x = [p[0] for p in way_points]
        y = [p[1] for p in way_points]
        plt.plot(x, y, linewidth=4)
    for p in circle_point:
        if math.isnan(p[0][0]):
            continue
        circle = patches.Circle(p[0], 1, color='r', zorder=3)
        axes.add_patch(circle)
    plt.show()


if __name__ == '__main__':
    map_dir = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/'
    map_name = "DR_USA_Roundabout_FT.osm"
    dataFile = 'D:/Dev/UCB task/Segmented_reference_path_DR_USA_Roundabout_FT.mat'
    data = scio.loadmat(dataFile)
    para_path = data['Segmented_reference_path']['para_path']
    circle_merge_point = data['Segmented_reference_path']['circle_merge_point'][0]
    plot_ref_path(map_dir+map_name, para_path, circle_merge_point)
