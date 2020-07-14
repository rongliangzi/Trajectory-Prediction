import openpyxl
import math
from align_ref_img import counterclockwise_rotate
import numpy as np
from utils import map_vis_without_lanelet
import matplotlib.pyplot as plt
from utils.roundabout_utils import nearest_c_point, on_press
import re
y_range = {0: [1050, 1020], 1: [1050, 1020], 6: [1050, 1010], 8: [1050, 1010], 19: [1050, 1010], 20: [1050, 1010]}
x_range = {2: [960, 985], 3: [], 15: [1030, 1065], 16: [1030, 1065], 17: [1065, 1100], 18: [1065, 1100]}
circle_range = {21: [(1054, 1014), (1066, 1025)], 22: [(1055, 1008), (1070, 1025)], 23: [(1070, 1025), (1078, 1017)],
                24: [(1066, 1025), (1083, 1007)], }


def read_funcs(func_file):
    func_d = dict()
    fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
    map_dir = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/'
    map_name = "DR_USA_Roundabout_EP.osm"
    map_file = map_dir + map_name
    map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
    img_id = 2
    with open(func_file) as f:
        lines = f.readlines()
        for line_id, line in enumerate(lines):
            func_d[line_id] = dict()
            if line_id != img_id:
                continue
            if line_id in [0, 1, 6, 8, 19, 20]:
                # x=f(y)
                coef = []
                p2 = re.compile('\((.*?)\)', re.S)
                a = re.findall(p2, line)
                for c in a:
                    coef.append(eval(c))
                p = np.poly1d(coef)
                yp = np.arange(y_range[line_id][0], y_range[line_id][1], (y_range[line_id][1]-y_range[line_id][0])/200)
                xp = p(yp)
                plt.text(xp[0], yp[0], 'start', fontsize=20)
                plt.text(xp[-1], yp[-1], 'end', fontsize=20)
                plt.plot(xp, yp, linewidth=2)
            elif line_id in [2, 3, 4, 5, 7, 10, 11, 15, 16, 17, 18]:
                # y=
                coef = []
                p2 = re.compile(r'[(](.*?)[)]', re.S)
                a = re.findall(p2, line)
                for c in a:
                    coef.append(eval(c))
                p = np.poly1d(coef)
                xp = np.arange(x_range[line_id][0], x_range[line_id][1], (x_range[line_id][1]-x_range[line_id][0])/200)
                yp = p(xp)
                plt.text(xp[0], yp[0], 'start', fontsize=20)
                plt.text(xp[-1], yp[-1], 'end', fontsize=20)
                plt.plot(xp, yp, linewidth=2)
            elif line_id in [21, 22, 23, 24]:
                # circle
                p1 = re.compile('\(x-(.*?)\)', re.S)
                a = re.findall(p1, line)
                circle_x = eval(a[0])
                p2 = re.compile('\(y-(.*?)\)', re.S)
                a = re.findall(p2, line)
                circle_y = eval(a[0])
                p2 = re.compile('=(.*)\^', re.S)
                a = re.findall(p2, line)
                circle_r = eval(a[0])
                points_x = [circle_x - circle_r]
                points_y = [circle_y]
                for i in range(359):
                    nx, ny = counterclockwise_rotate(circle_x - circle_r, circle_y, (circle_x, circle_y),
                                                     (i + 1) * math.pi / 180)
                    points_x.append(nx)
                    points_y.append(ny)
                cp_s = circle_range[line_id][0]
                cp_e = circle_range[line_id][1]
                s_i, _, _ = nearest_c_point(cp_s, points_x, points_y)
                e_i, _, _ = nearest_c_point(cp_e, points_x, points_y)
                if e_i < s_i:
                    xp = points_x[s_i:] + points_x[:e_i]
                    yp = points_y[s_i:] + points_y[:e_i]
                else:
                    xp = points_x[s_i: e_i]
                    yp = points_y[s_i: e_i]
                xp = [cp_s[0]] + xp + [cp_e[0]]
                yp = [cp_s[1]] + yp + [cp_e[1]]
                plt.plot(xp, yp, linewidth=2)
                plt.text(xp[0], yp[0], 'start', fontsize=20)
                plt.text(xp[-1], yp[-1], 'end', fontsize=20)
                plt.plot(xp, yp, linewidth=2, zorder=30)
            else:  # [9, 12, 13, 14]
                pass
    # fig.canvas.mpl_connect('button_press_event', on_press)
    # plt.show()
    plt.savefig('D:/Dev/UCB task/path_imgs/EP/{}.png'.format(img_id))
