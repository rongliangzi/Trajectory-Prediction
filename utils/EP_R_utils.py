import openpyxl
import math
from align_ref_img import counterclockwise_rotate
import numpy as np
from utils import map_vis_without_lanelet
import matplotlib.pyplot as plt
from utils.roundabout_utils import nearest_c_point, on_press
from utils.intersection_utils import cal_dis
import re
from utils.roundabout_utils import get_circle
y_range = {0: [1025.27, 1044.25], 1: (1044.513, 976.992), 6: (1005.624, 989.106), 8: (980.936, 1008.159),
           10: (1012.872, 1002.185), 11: (998.668, 1025.163)}
x_range = {2: (1014.832, 965.23), 3: (960.058, 971.693), 4: (990.483, 968.049), 5: (968.009, 1015.247),
           7: (972.215, 987.798)}
# y range of ellipse
ellipse_range = {9: [1012.93, 1001.84], 12: [1018.4, 1020.09], 13: [998.47, 1007.31], 14: [1002, 989]}
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
path_mapping = {'1-4': '1-4', '1': '1-6', '1-9-5': '1-9', '3-1-4': '3-4', '3-1': '3-6', '3-1-9-5': '3-9',
                '5-11-0': '5-0', '5-11-2': '5-2', '6-1': '5-6', '5': '5-9', '7-11-0': '7-0', '7-11-2': '7-2',
                '7-11-4': '7-4', '7-13-5': '7-9', '8-5': '8-9', '2-0': '10-0', '2': '10-2', '2-12-4': '10-4',
                '2-12-4-10-14-1': '10-6'}
r_paths = list(path_mapping.keys())


def read_funcs(func_file, wp_n=200):
    map_dir = 'D:/Downloads/INTERACTION-Dataset-DR-v1_0/maps/'
    map_name = "DR_USA_Roundabout_EP.osm"
    map_file = map_dir + map_name
    div_path_points = dict()
    with open(func_file) as f:
        lines = f.readlines()
        for line_id, line in enumerate(lines):
            if line_id in y_range.keys():  #
                fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
                map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
                # x=f(y)
                coef = []
                p2 = re.compile('\((.*?)\)', re.S)
                a = re.findall(p2, line)
                for c in a[:-2]:
                    coef.append(eval(c))
                p = np.poly1d(coef)
                yp = np.arange(y_range[line_id][0], y_range[line_id][1], (y_range[line_id][1]-y_range[line_id][0])/wp_n)
                xp = p(yp)
                plt.text(xp[0], yp[0], 'start', fontsize=20)
                plt.text(xp[-1], yp[-1], 'end', fontsize=20)
                plt.plot(xp, yp, linewidth=2)
                plt.savefig('D:/Dev/TrajPred/path_imgs/EP/div_{}.png'.format(line_id))
                plt.close()
                div_path_points[line_id] = np.array([[x, y] for x, y in zip(xp, yp)])
            elif line_id in x_range.keys():  #
                fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
                map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
                # y=
                coef = []
                p2 = re.compile(r'[(](.*?)[)]', re.S)
                a = re.findall(p2, line)
                for c in a[:-2]:
                    coef.append(eval(c))
                p = np.poly1d(coef)
                xp = np.arange(x_range[line_id][0], x_range[line_id][1], (x_range[line_id][1]-x_range[line_id][0])/wp_n)
                yp = p(xp)
                plt.text(xp[0], yp[0], 'start', fontsize=20)
                plt.text(xp[-1], yp[-1], 'end', fontsize=20)
                plt.plot(xp, yp, linewidth=2)
                plt.savefig('D:/Dev/TrajPred/path_imgs/EP/div_{}.png'.format(line_id))
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
                plt.savefig('D:/Dev/TrajPred/path_imgs/EP/div_{}.png'.format(line_id))
                plt.close()
                div_path_points[line_id] = np.array([[x, y] for x, y in zip(xp, yp)])
        ref_path_points = get_r_path(div_path_points, map_file)
    ref_path_id_points = dict()
    for k in ref_path_points.keys():
        ref_path_id_points[path_mapping[k]] = ref_path_points[k]
    return ref_path_id_points


def get_insert_circle(v1, v2, v3, c_insert_k):
    x0, y0, r = get_circle(v1, v2, v3)
    theta_p = math.atan2(v1[1] - y0, v1[0] - x0)
    if theta_p < 0:
        theta_p += math.pi * 2
    theta_c = math.atan2(v2[1] - y0, v2[0] - x0)
    if theta_c < 0:
        theta_c += math.pi * 2
    if theta_c < theta_p:
        theta_c += math.pi*2
    p1 = []
    for j in range(1, c_insert_k):
        x_rot, y_rot = counterclockwise_rotate(v1[0], v1[1], (x0, y0),
                                               (theta_c - theta_p) * j / c_insert_k)
        p1.append([x_rot, y_rot])
    p1 = np.array(p1)

    theta_n = math.atan2(v3[1] - y0, v3[0] - x0)
    if theta_n < 0:
        theta_n += math.pi * 2
    if theta_n < theta_c:
        theta_n += math.pi * 2
    p2 = []
    for j in range(1, c_insert_k):
        x_rot, y_rot = counterclockwise_rotate(v2[0], v2[1], (x0, y0),
                                               (theta_n - theta_c) * j / c_insert_k)
        p2.append([x_rot, y_rot])
    p2 = np.array(p2)
    return p1, p2


def get_r_path(div_path_points, map_file):
    ref_path_points = dict()
    for ref_path in r_paths:
        # if ref_path != '2-12-4':
        #     continue
        fig, axes = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
        map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, 0, 0)
        div_paths = ref_path.split('-')
        if len(div_paths) == 1:
            ref_path_points[ref_path] = div_path_points[int(div_paths[0])].copy()
        elif len(div_paths) == 2:
            ref_path_data0 = div_path_points[int(div_paths[0])].copy()
            ref_path_data1 = div_path_points[int(div_paths[1])].copy()
            dis1 = cal_dis(ref_path_data0, ref_path_data1)
            min_id1 = np.argmin(dis1)
            i1, j1 = min_id1 // dis1.shape[1], min_id1 % dis1.shape[1]
            ref_path_data0 = ref_path_data0[:i1]
            ref_path_data1 = ref_path_data1[j1:]
            if ref_path == '1-4':
                ref_path_data1[0] = (ref_path_data0[-1]+ref_path_data1[1])/2
                ref_path_data1[1] = (ref_path_data1[0] + ref_path_data1[2]) / 2
            elif ref_path == '3-1':
                ref_path_data1[0] = (ref_path_data0[-1] + ref_path_data1[1]) / 2
            elif ref_path == '6-1':
                ref_path_data1[:, 0] -= 0.1
                ref_path_data1[0] = (ref_path_data0[-1] + ref_path_data1[1]) / 2
            elif ref_path == '8-5':
                p = [1003.03, 1008.26]
                p1, p2 = get_insert_circle(ref_path_data1[8], p, ref_path_data0[-2], 10)
                ref_path_data0 = ref_path_data0[:-1]
                ref_path_data1 = np.vstack((p2[::-1], np.array(p), p1[::-1], ref_path_data1[8:]))
            elif ref_path == '2-0':
                ref_path_data1[:, 1] -= 0.12
            ref_path_points[ref_path] = np.vstack((ref_path_data0, ref_path_data1))
        elif len(div_paths) == 4:
            ref_path_data0 = div_path_points[int(div_paths[0])].copy()
            ref_path_data1 = div_path_points[int(div_paths[1])].copy()
            ref_path_data2 = div_path_points[int(div_paths[2])].copy()
            ref_path_data3 = div_path_points[int(div_paths[3])].copy()
            dis1 = cal_dis(ref_path_data0, ref_path_data1)
            min_id1 = np.argmin(dis1)
            i1, j1 = min_id1 // dis1.shape[1], min_id1 % dis1.shape[1]
            ref_path_data0 = ref_path_data0[:i1]

            dis2 = cal_dis(ref_path_data1, ref_path_data2)
            min_id2 = np.argmin(dis2)
            i2, j2 = min_id2 // dis2.shape[1], min_id2 % dis2.shape[1]
            ref_path_data1 = ref_path_data1[j1:i2]

            dis3 = cal_dis(ref_path_data2, ref_path_data3)
            min_id3 = np.argmin(dis3)
            i3, j3 = min_id3 // dis3.shape[1], min_id3 % dis3.shape[1]
            ref_path_data2 = ref_path_data2[j2:i3]

            ref_path_data3 = ref_path_data3[j3:]
            if ref_path == '3-1-9-5':
                ref_path_data1[0] = (ref_path_data1[1] + ref_path_data0[-1])/2
                ref_path_data3[:, 1] += 0.35
                ref_path_data3 = ref_path_data3[4:]
                p = [982, 1001.8]
                p1, p2 = get_insert_circle(ref_path_data2[-1], p, ref_path_data3[0], 10)
                ref_path_data3 = np.vstack((p1, np.array(p), p2, ref_path_data3))

            ref_path_points[ref_path] = np.vstack((ref_path_data0, ref_path_data1, ref_path_data2,
                                                   ref_path_data3))
            plt.plot([p[0] for p in ref_path_data0], [p[1] for p in ref_path_data0], c='r')
            plt.plot([p[0] for p in ref_path_data1], [p[1] for p in ref_path_data1], c='g')
            plt.plot([p[0] for p in ref_path_data2], [p[1] for p in ref_path_data2], c='b')
            plt.plot([p[0] for p in ref_path_data3], [p[1] for p in ref_path_data3], c='r')
        elif len(div_paths) == 6:
            ref_path_data0 = div_path_points[int(div_paths[0])].copy()
            ref_path_data1 = div_path_points[int(div_paths[1])].copy()
            ref_path_data2 = div_path_points[int(div_paths[2])].copy()
            ref_path_data3 = div_path_points[int(div_paths[3])].copy()
            ref_path_data4 = div_path_points[int(div_paths[4])].copy()
            ref_path_data5 = div_path_points[int(div_paths[5])].copy()
            dis1 = cal_dis(ref_path_data0, ref_path_data1)
            min_id1 = np.argmin(dis1)
            i1, j1 = min_id1 // dis1.shape[1], min_id1 % dis1.shape[1]
            ref_path_data0 = ref_path_data0[:i1]

            dis2 = cal_dis(ref_path_data1, ref_path_data2)
            min_id2 = np.argmin(dis2)
            i2, j2 = min_id2 // dis2.shape[1], min_id2 % dis2.shape[1]
            ref_path_data1 = ref_path_data1[j1:i2]

            dis3 = cal_dis(ref_path_data2, ref_path_data3)
            min_id3 = np.argmin(dis3)
            i3, j3 = min_id3 // dis3.shape[1], min_id3 % dis3.shape[1]
            ref_path_data2 = ref_path_data2[j2:i3]

            dis4 = cal_dis(ref_path_data3, ref_path_data4)
            min_id4 = np.argmin(dis4)
            i4, j4 = min_id4 // dis4.shape[1], min_id4 % dis4.shape[1]
            ref_path_data3 = ref_path_data3[j3:i4]

            dis5 = cal_dis(ref_path_data4, ref_path_data5)
            min_id5 = np.argmin(dis5)
            i5, j5 = min_id5 // dis5.shape[1], min_id5 % dis5.shape[1]
            ref_path_data0[:, 1] -= 0.017
            ref_path_data4 = ref_path_data4[j4:i5]

            ref_path_data5 = ref_path_data5[j5:]
            ref_path_data2 = np.vstack((ref_path_data2[:-1], np.array([976.3740876224332, 1013.0162002656417])))
            ref_path_data4[:, 0] -= 0.05
            ref_path_data5[:, 0] -= 0.037
            p = [976.480695941081, 1013.138906751151]
            p1, p2 = get_insert_circle(ref_path_data2[-4], p, ref_path_data3[0], 10)
            ref_path_data2 = ref_path_data2[:-3]
            ref_path_data3 = np.vstack((p1, np.array(p), p2, ref_path_data3))
            plt.plot([p[0] for p in ref_path_data0], [p[1] for p in ref_path_data0], c='r')
            plt.plot([p[0] for p in ref_path_data1], [p[1] for p in ref_path_data1], c='g')
            plt.plot([p[0] for p in ref_path_data2], [p[1] for p in ref_path_data2], c='b', marker='x')
            plt.plot([p[0] for p in ref_path_data3], [p[1] for p in ref_path_data3], c='r', marker='x')
            plt.plot([p[0] for p in ref_path_data4], [p[1] for p in ref_path_data4], c='g')
            plt.plot([p[0] for p in ref_path_data5], [p[1] for p in ref_path_data5], c='b')
            ref_path_points[ref_path] = np.vstack((ref_path_data0, ref_path_data1, ref_path_data2,
                                                   ref_path_data3, ref_path_data4, ref_path_data5))
        else:
            ref_path_data0 = div_path_points[int(div_paths[0])].copy()
            ref_path_data1 = div_path_points[int(div_paths[1])].copy()

            ref_path_data2 = div_path_points[int(div_paths[2])].copy()
            dis1 = cal_dis(ref_path_data0, ref_path_data1)
            dis2 = cal_dis(ref_path_data1, ref_path_data2)
            min_id1 = np.argmin(dis1)
            i1, j1 = min_id1 // dis1.shape[1], min_id1 % dis1.shape[1]
            min_id2 = np.argmin(dis2)
            i2, j2 = min_id2 // dis2.shape[1], min_id2 % dis2.shape[1]
            ref_path_data0 = ref_path_data0[:i1]
            ref_path_data1 = ref_path_data1[j1:i2 + 1]
            ref_path_data2 = ref_path_data2[j2:]
            if ref_path == '1-9-5':
                ref_path_data2[:, 1] += 0.4
                ref_path_data2 = ref_path_data2[4:]
                p = [982, 1001.8]
                p1, p2 = get_insert_circle(ref_path_data1[-1], p, ref_path_data2[0], 10)
                ref_path_data2 = np.vstack((p1, np.array(p), p2, ref_path_data2))
            elif ref_path == '3-1-4':
                ref_path_data1[0] = (ref_path_data0[-1] + ref_path_data1[1]) / 2
                ref_path_data2 = ref_path_data2[1:]
                p = [976.23, 1012.95]
                p1, p2 = get_insert_circle(ref_path_data2[1], p, ref_path_data1[-1], 10)
                ref_path_data2 = np.vstack((p2[::-1], np.array(p), p1[::-1], ref_path_data2[1:]))
            elif ref_path == '5-11-2':
                p = [993.67, 1004.37]
                p1, p2 = get_insert_circle(ref_path_data0[-1], p, ref_path_data1[1], 10)
                ref_path_data1 = np.vstack((p1, np.array(p), p2, ref_path_data1[1:]))
                ref_path_data2 = ref_path_data2[1:]
            elif ref_path == '7-11-2':
                ref_path_data0[:, 1] += 0.05
                ref_path_data2 = ref_path_data2[1:]
                ref_path_data2[:, 1] -= 0.05
            elif ref_path == '7-11-4':
                ref_path_data0[:, 1] += 0.05
                p = [990.64, 1019.67]
                p1, p2 = get_insert_circle(ref_path_data1[-4], p, ref_path_data2[1], 10)
                ref_path_data1 = ref_path_data1[:-4]
                ref_path_data2 = np.vstack((p1, np.array(p), p2, ref_path_data2[2:]))
            elif ref_path == '7-13-5':
                ref_path_data1[:, 1] += 0.01
                ref_path_data2[:, 1] += 0.032
                ref_path_data2 = ref_path_data2[1:]
            elif ref_path == '5-11-0':
                p = [993.67, 1004.37]
                p1, p2 = get_insert_circle(ref_path_data0[-1], p, ref_path_data1[1], 10)
                ref_path_data1 = np.vstack((p1, np.array(p), p2, ref_path_data1[1:-1]))
                ref_path_data2[:, 1] -= 0.19
            elif ref_path == '7-11-0':
                ref_path_data0[:, 1] += 0.05
                ref_path_data2[:, 1] -= 0.19
                ref_path_data1[-1] = (ref_path_data1[-2] + ref_path_data2[0]) / 2
            elif ref_path == '2-12-4':
                ref_path_data0[:, 1] -= 0.017
            plt.plot([p[0] for p in ref_path_data0], [p[1] for p in ref_path_data0], c='r')
            plt.plot([p[0] for p in ref_path_data1], [p[1] for p in ref_path_data1], c='g')
            plt.plot([p[0] for p in ref_path_data2], [p[1] for p in ref_path_data2], c='b')
            ref_path_points[ref_path] = np.vstack((ref_path_data0, ref_path_data1, ref_path_data2))
        xp, yp = [point[0] for point in ref_path_points[ref_path]], [point[1] for point in ref_path_points[ref_path]]
        xy_p = np.array([[x, y] for x, y in zip(xp, yp)])
        plt.text(xp[0], yp[0], 'start', fontsize=20)
        plt.text(xp[-1], yp[-1], 'end', fontsize=20)
        # plt.plot(xp, yp, linewidth=1, zorder=30, marker='x')
        plt.title(ref_path)
        # fig.canvas.mpl_connect('button_press_event', on_press)
        # plt.show()
        plt.savefig('D:/Dev/TrajPred/path_imgs/EP/{}.png'.format(ref_path))
        plt.close()
        ref_path_points[ref_path] = xy_p
    return ref_path_points
