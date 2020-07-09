import openpyxl
import math
from align_ref_img import counterclockwise_rotate
import numpy as np
from utils import map_vis_without_lanelet
import matplotlib.pyplot as plt
from utils.roundabout_utils import nearest_c_point


def read_funcs(func_file):
    func_d = dict()
    wb = openpyxl.load_workbook(func_file)
    sheet = wb['Sheet1']
    fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
    map_dir = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/'
    map_name = "DR_USA_Roundabout_EP.osm"
    map_file = map_dir + map_name
    map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
    for r in range(2, sheet.max_row+1):
        path_id = r-2
        func_d[path_id] = dict()
        flag = sheet.cell(row=r, column=2).value
        if flag == 'N/A':
            circle_x = sheet.cell(row=r, column=8).value
            circle_y = sheet.cell(row=r, column=9).value
            circle_r = sheet.cell(row=r, column=10).value
            points_x = [circle_x - circle_r]
            points_y = [circle_y]
            for i in range(359):
                nx, ny = counterclockwise_rotate(circle_x - circle_r, circle_y, (circle_x, circle_y),
                                                 (i + 1) * math.pi / 180)
                points_x.append(nx)
                points_y.append(ny)
            if path_id == 21:
                start_x = 1066
                start_y = 1025
                _, csx, csy = nearest_c_point((start_x, start_y), points_x, points_y)
                points_x = [csx]
                points_y = [csy]
                end_x = 1083
                end_y = 1007
                _, cex, cey = nearest_c_point((end_x, end_y), points_x, points_y)
            elif path_id == 20:
                points_x = [1070]
                points_y = [1025]
                end_x = 1080
                end_y = 1007
            elif path_id == 19:
                points_x = [1055]
                points_y = [1008]
                end_x = 1070
                end_y = 1025
            elif path_id == 18:
                points_x = [1055]
                points_y = [1008]
                end_x = 1070
                end_y = 1025
            for i in range(359):
                nx, ny = counterclockwise_rotate(circle_x - circle_r, circle_y, (circle_x, circle_y),
                                                 (i + 1) * math.pi / 180)
                if nx > end_x:
                    break
                points_x.append(nx)
                points_y.append(ny)
            plt.plot(points_x, points_y, linewidth=4)
        else:
            a = sheet.cell(row=r, column=3).value
            b = sheet.cell(row=r, column=4).value
            c = sheet.cell(row=r, column=5).value
            d = sheet.cell(row=r, column=6).value
            e = sheet.cell(row=r, column=7).value
            if flag == 0:
                p = np.poly1d([a, b, c, d, e])
                xp = np.arange(1030, 1100, 0.2)
                yp = p(xp)

                plt.text(xp[0], yp[0], 'start', fontsize=20)
                plt.text(xp[-1], yp[-1], 'end', fontsize=20)
                plt.plot(xp, yp, linewidth=4)
    plt.show()
