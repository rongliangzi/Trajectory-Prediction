import openpyxl
import math
from align_ref_img import counterclockwise_rotate
import numpy as np
from utils import map_vis_without_lanelet
import matplotlib.pyplot as plt
from utils.roundabout_utils import nearest_c_point, on_press
from utils.intersection_utils import cal_dis
import re
y_range = {0: [1050, 1020], 1: [1050, 1020], 6: [989, 1006], 8: [980, 1008], 19: [1050, 1025], 20: [1027, 1050]}
x_range = {2: [960, 985], 3: [960, 975], 4: [965, 998], 5: [965, 1010], 7: [970, 1000], 10: [960, 1010],
           11: [960, 1010], 15: [1090, 1080], 16: [1030, 1095], 17: [1083, 1100], 18: [1100, 1035]}
circle_range = {21: [(1054, 1014), (1066, 1025)], 22: [(1055, 1008), (1072, 1027)], 23: [(1070, 1025), (1078, 1017)],
                24: [(1066, 1025), (1083, 1007)], }
# y range of ellipse
ellipse_range = {9: [990, 1020], 12: [990, 1030], 13: [990, 1030], 14: [990, 1030]}
# it_paths = ['16-22-20', '19-21-18', '19-24-17', '15-23-20', '18', '16']
it_paths = []
ellipse_c = {9: [16435240000, - 7063820000, -252551490054.81, 5291401216742.08, 9838149580819350],
             12: [1.3019690089539376e-09, 6.832444032549981e-08, - 7.202131029678265e-05,  8.964949222166735e-06, 1],
             13: [3.951677128103497e-05, 9.132466734761502e-05, - 0.0012877825475609331,  0.014831764277340445, 1],
             14: [0.0000004449,  0.0000004781, - 0.0001326617, - 0.0007239847, 1]}


def read_funcs(func_file, wp_n=200):
    map_dir = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/'
    map_name = "DR_USA_Roundabout_EP.osm"
    map_file = map_dir + map_name
    div_path_points = dict()
    with open(func_file) as f:
        lines = f.readlines()
        for line_id, line in enumerate(lines):
            # if line_id != img_id:
            #     continue
            if line_id in [0, 1, 6, 8, 19, 20]:  #
                fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
                map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
                # x=f(y)
                coef = []
                p2 = re.compile('\((.*?)\)', re.S)
                a = re.findall(p2, line)
                for c in a:
                    coef.append(eval(c))
                p = np.poly1d(coef)
                yp = np.arange(y_range[line_id][0], y_range[line_id][1], (y_range[line_id][1]-y_range[line_id][0])/wp_n)
                xp = p(yp)
                plt.text(xp[0], yp[0], 'start', fontsize=20)
                plt.text(xp[-1], yp[-1], 'end', fontsize=20)
                plt.plot(xp, yp, linewidth=2)
                plt.savefig('D:/Dev/UCB task/path_imgs/EP/div_{}.png'.format(line_id))
                plt.close()
                div_path_points[line_id] = np.array([[x, y] for x, y in zip(xp, yp)])
            elif line_id in [2, 3, 4, 5, 7, 10, 11, 15, 16, 17, 18]:  #
                fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
                map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
                # y=
                coef = []
                p2 = re.compile(r'[(](.*?)[)]', re.S)
                a = re.findall(p2, line)
                for c in a:
                    coef.append(eval(c))
                p = np.poly1d(coef)
                xp = np.arange(x_range[line_id][0], x_range[line_id][1], (x_range[line_id][1]-x_range[line_id][0])/wp_n)
                yp = p(xp)
                plt.text(xp[0], yp[0], 'start', fontsize=20)
                plt.text(xp[-1], yp[-1], 'end', fontsize=20)
                plt.plot(xp, yp, linewidth=2)
                plt.savefig('D:/Dev/UCB task/path_imgs/EP/div_{}.png'.format(line_id))
                plt.close()
                div_path_points[line_id] = np.array([[x, y] for x, y in zip(xp, yp)])
            elif line_id in [21, 22, 23, 24]:
                fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
                map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
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
                # xp = [cp_s[0]] + xp + [cp_e[0]]
                # yp = [cp_s[1]] + yp + [cp_e[1]]
                plt.plot(xp, yp, linewidth=2)
                plt.text(xp[0], yp[0], 'start', fontsize=20)
                plt.text(xp[-1], yp[-1], 'end', fontsize=20)
                plt.plot(xp, yp, linewidth=2, zorder=30)
                plt.savefig('D:/Dev/UCB task/path_imgs/EP/div_{}.png'.format(line_id))
                plt.close()
                if line_id in [21, 23]:
                    xp = xp[::-1]
                    yp = yp[::-1]
                div_path_points[line_id] = np.array([[x, y] for x, y in zip(xp, yp)])
            else:  # [9, 12, 13, 14]
                coef = ellipse_c[line_id]
                yp = np.arange(ellipse_range[line_id][0], ellipse_range[line_id][1],
                               (ellipse_range[line_id][1]-ellipse_range[line_id][0])/wp_n)
                xp = []
                for y in yp:
                    args = [coef[0], coef[1]*y, coef[2]*y**2+coef[3]*y+coef[4]]
                    roots = np.roots(args)
                    x = 0
                    for r in roots:
                        if 960 < r < 1000:
                            x = r
                    xp.append(x)
                fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
                map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
                plt.text(xp[0], yp[0], 'start', fontsize=20)
                plt.text(xp[-1], yp[-1], 'end', fontsize=20)
                plt.plot(xp, yp, linewidth=2)
                plt.savefig('D:/Dev/UCB task/path_imgs/EP/div_{}.png'.format(line_id))
                plt.close()
                div_path_points[line_id] = np.array([[x, y] for x, y in zip(xp, yp)])
        get_it_path(div_path_points, map_file)


def get_it_path(div_path_points, map_file):
    ref_path_points = dict()
    for ref_path in it_paths:
        fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
        map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)

        div_paths = ref_path.split('-')
        if len(div_paths) == 1:
            ref_path_points[ref_path] = div_path_points[int(div_paths[0])]
        else:
            ref_path_data0 = div_path_points[int(div_paths[0])]
            ref_path_data1 = div_path_points[int(div_paths[1])]
            ref_path_data2 = div_path_points[int(div_paths[2])]

            dis1 = cal_dis(ref_path_data0, ref_path_data1)
            dis2 = cal_dis(ref_path_data1, ref_path_data2)
            min_id1 = np.argmin(dis1)
            i1, j1 = min_id1 // dis1.shape[1], min_id1 % dis1.shape[1]
            min_id2 = np.argmin(dis2)
            i2, j2 = min_id2 // dis2.shape[1], min_id2 % dis2.shape[1]
            ref_path_data0 = ref_path_data0[:i1]
            ref_path_data1 = ref_path_data1[j1:i2 + 1]
            ref_path_data2 = ref_path_data2[j2:]
            ref_path_points[ref_path] = np.vstack((ref_path_data0, ref_path_data1, ref_path_data2))
        xp, yp = [point[0] for point in ref_path_points[ref_path]], [point[1] for point in ref_path_points[ref_path]]
        plt.plot(xp, yp, linewidth=2)
        plt.text(xp[0], yp[0], 'start', fontsize=20)
        plt.text(xp[-1], yp[-1], 'end', fontsize=20)
        plt.plot(xp, yp, linewidth=2, zorder=30)
        # fig.canvas.mpl_connect('button_press_event', on_press)
        # plt.show()
        plt.savefig('D:/Dev/UCB task/path_imgs/EP/{}.png'.format(ref_path))
        plt.close()
