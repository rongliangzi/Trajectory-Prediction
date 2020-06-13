import matplotlib.pyplot as plt
from utils import map_vis_without_lanelet



def plot_starting_area(map_file):
    fig, axes = plt.subplots(1, 1)
    map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
    for key, v in starting_area_dict.items():
        x = v['x']
        y = v['y']
        plt.plot(x[0:2], y[0:2], c='r', zorder=40)
        plt.plot(x[1:3], y[1:3], c='r', zorder=40)
        plt.plot(x[2:4], y[2:4], c='r', zorder=40)
        plt.plot(x[3:] + x[0:1], y[3:] + y[0:1], c='r', zorder=40)
    plt.show()


if __name__ == '__main__':
    map_dir = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/'
    map_name = "DR_USA_Roundabout_FT.osm"
    plot_starting_area(map_dir + map_name)
