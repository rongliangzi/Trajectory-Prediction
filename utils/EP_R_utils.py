import openpyxl
import math
from align_ref_img import counterclockwise_rotate
import numpy as np
from utils import map_vis_without_lanelet
import matplotlib.pyplot as plt
from utils.roundabout_utils import nearest_c_point, on_press
from utils.intersection_utils import cal_dis
import re
y_range = {0: [1044.25, 1025.27], 1: (1044.513, 976.992), 6: (1005.624, 989.106), 8: (980.936, 1008.159),
           10: (1012.872, 1002.185), 11: (998.668, 1025.163)}
x_range = {2: (1014.832, 965.23), 3: (960.058, 971.693), 4: (990.483, 968.049), 5: (968.009, 1015.247),
           7: (972.215, 987.798)}
# y range of ellipse
ellipse_range = {9: [1001.84, 1012.93], 12: [1018.4, 1020.09], 13: [998.47, 1007.31], 14: [989, 1002]}
ellipse_c = {9: [245966132940052094389069248187827666499131677473767424,
                 -105715553237103548864463554356788343246367354622312448,
                 -377962922214871373029773089062947781303388459621406474240,
                 11359061980039628596743822436388178607994751391105024,
                 79189929390523228544261997509007647019017798340149510144,
                 147235550419699888248387563756411216617339302179518598951749],
             12: [64825022356445940467230891397667536979961577472,
                  3401873117664108950488191217152568392209695506432,
                  -3585940232080164897836182762064124589087140538744832,
                  44630685374283474877017875878279149827842845442048,
                  - 94286664721245006117141227480933928714491887282552832,
                  49789988786698826227545791881560777988210019369284852475],
             13: [207677689661223658512495408704363840933524756076953600,
                  -479950545274804687428522812029984988407425874704465920,
                  67678531315601681457838429480809277593968449022836342784,
                  277295705529740872254398145127378781092298236578758656,
                  - 77947323095098243233117663067298195029597788054841458688,
                  5255431628871285618305630467260175326979340889080391702465],
             14: [15105450864250551579164303389419958109438296352358400,
                  16232297529287823928064678548405889504888435591086080,
                  - 45045496042619243309287321719657419605423757175820386304,
                  4360801366460841086327853451695656639142519713562624,
                  - 24583024456398024193316492369435999541490736235246780416,
                  33955171138598393492319267315978121325342817422499012077171]}
signs = {9: 2, 12: 1, 13: 2, 14: 1}


def read_funcs(func_file, wp_n=200):
    map_dir = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/'
    map_name = "DR_USA_Roundabout_EP.osm"
    map_file = map_dir + map_name
    div_path_points = dict()
    with open(func_file) as f:
        lines = f.readlines()
        for line_id, line in enumerate(lines):
            # if line_id not in [9, 12, 13, 14, 10, 11]:
            #     continue
            if line_id in y_range.keys():  #
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
            elif line_id in x_range.keys():  #
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
            elif line_id in ellipse_range.keys():  # [9, 12, 13, 14]
                coef = ellipse_c[line_id]
                yp = np.arange(ellipse_range[line_id][0], ellipse_range[line_id][1],
                               (ellipse_range[line_id][1]-ellipse_range[line_id][0])/wp_n)
                xp = []
                for y in yp:
                    a, b, c = coef[0], coef[1]*y+coef[2], coef[3]*y**2+coef[4]*y+coef[5]
                    x = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a) if signs[line_id] == 1 else (-b - (
                                b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
                    xp.append(x)
                fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
                map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
                plt.text(xp[0], yp[0], 'start', fontsize=20)
                plt.text(xp[-1], yp[-1], 'end', fontsize=20)
                plt.plot(xp, yp, linewidth=2)
                plt.savefig('D:/Dev/UCB task/path_imgs/EP/div_{}.png'.format(line_id))
                plt.close()
                div_path_points[line_id] = np.array([[x, y] for x, y in zip(xp, yp)])
        # get_it_path(div_path_points, map_file)


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
