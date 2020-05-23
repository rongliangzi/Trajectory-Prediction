import matplotlib.pyplot as plt
from utils import map_vis_without_lanelet
import os


starting_area_dict = dict()
starting_area_dict[2] = dict()
starting_area_dict[2]['x'] = [992, 992.2, 1002.3, 1002]
starting_area_dict[2]['y'] = [1005.5, 1002.5, 1002, 1005.5]
starting_area_dict[2]['stoplinex'] = [1000.1, 1000.2]
starting_area_dict[2]['stopliney'] = [1005.5, 1002.5]
starting_area_dict[3] = dict()
starting_area_dict[3]['x'] = [992, 992.2, 1002.3, 1002]
starting_area_dict[3]['y'] = [1002.5, 997.5, 993, 1002]
starting_area_dict[3]['stoplinex'] = [1000.15, 1000.3]
starting_area_dict[3]['stopliney'] = [1001.5, 997]
starting_area_dict[6] = dict()
starting_area_dict[6]['x'] = [1024, 1025.5, 1028.9, 1027.6]
starting_area_dict[6]['y'] = [996.5, 986, 986.5, 997]
starting_area_dict[6]['stoplinex'] = [1024, 1028]
starting_area_dict[6]['stopliney'] = [994.6, 994.8]
starting_area_dict[7] = dict()
starting_area_dict[7]['x'] = [1027.9, 1029.2, 1036.8, 1038.3]
starting_area_dict[7]['y'] = [997.5, 986.5, 987, 998]
starting_area_dict[7]['stoplinex'] = [1028.8, 1036]
starting_area_dict[7]['stopliney'] = [994.8, 995]
starting_area_dict[9] = dict()
starting_area_dict[9]['x'] = [1031.5, 1032.5, 1042.5, 1042.3]
starting_area_dict[9]['y'] = [1017, 1005.5, 1006, 1010]
starting_area_dict[9]['stoplinex'] = [1033.3, 1034.3]
starting_area_dict[9]['stopliney'] = [1014, 1006]
starting_area_dict[12] = dict()
starting_area_dict[12]['x'] = [1015.3, 1015.5, 1020.5, 1019]
starting_area_dict[12]['y'] = [1020, 1010.5, 1010.4, 1020.2]
starting_area_dict[12]['stoplinex'] = [1016.5, 1020]
starting_area_dict[12]['stopliney'] = [1012, 1012]
starting_area_dict[13] = dict()
starting_area_dict[13]['x'] = [1011.5, 1011.7, 1016, 1015.5]
starting_area_dict[13]['y'] = [1020, 1010.5, 1010.4, 1020.2]
starting_area_dict[13]['stoplinex'] = [1011.7, 1016.3]
starting_area_dict[13]['stopliney'] = [1012, 1012]
starting_area_dict[14] = dict()
starting_area_dict[14]['x'] = [1006, 996.2, 1011.4, 1011.1]
starting_area_dict[14]['y'] = [1020, 1010, 1009.9, 1020.2]
starting_area_dict[14]['stoplinex'] = [1004, 1011.3]
starting_area_dict[14]['stopliney'] = [1012, 1012]


def find_starting_area(work_dir):
    d = 8  # fig size
    r = 100  # range of x and y
    fig, axes = plt.subplots(1, 1, figsize=(d, d), dpi=200)
    lanelet_map_file = "D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/DR_USA_Intersection_MA.osm"
    map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, 0, 0)
    x = [1006, 996.2, 1011.4, 1011.1]
    y = [1019, 1010, 1009.9, 1019.2]

    plt.plot(x[0:2], y[0:2], c='r', zorder=40)
    plt.plot(x[1:3], y[1:3], c='r', zorder=40)
    plt.plot(x[2:4], y[2:4], c='r', zorder=40)
    plt.plot(x[3:]+x[0:1], y[3:]+y[0:1], c='r', zorder=40)
    xs, ys = 980, 950
    plt.xlim(xs, xs + r)
    plt.ylim(ys, ys + r)
    # remove the white biankuang
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    save_dir = work_dir + 'global_images/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.savefig(save_dir+'starting_area.png')
    plt.close()


def plot_starting_area(work_dir):
    d = 8  # fig size
    r = 100  # range of x and y
    fig, axes = plt.subplots(1, 1, figsize=(d, d), dpi=200)
    lanelet_map_file = "D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/DR_USA_Intersection_MA.osm"
    map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, 0, 0)
    for key, v in starting_area_dict.items():
        if 'stoplinex' in v:
            plt.plot(v['stoplinex'], v['stopliney'], zorder=40, c='g')
            k = (v['stopliney'][1]-v['stopliney'][0])/(v['stoplinex'][1]-v['stoplinex'][0])
            b = (v['stoplinex'][1]*v['stopliney'][0]-v['stoplinex'][0]*v['stopliney'][1])/(v['stoplinex'][1]-v['stoplinex'][0])
        x = v['x']
        y = v['y']
        plt.plot(x[0:2], y[0:2], c='r', zorder=40)
        plt.plot(x[1:3], y[1:3], c='r', zorder=40)
        plt.plot(x[2:4], y[2:4], c='r', zorder=40)
        plt.plot(x[3:] + x[0:1], y[3:] + y[0:1], c='r', zorder=40)
    xs, ys = 980, 950
    plt.xlim(xs, xs + r)
    plt.ylim(ys, ys + r)
    # remove the white biankuang
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    save_dir = work_dir + 'global_images/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.savefig(save_dir+'all_starting_area.png')
    plt.close()
