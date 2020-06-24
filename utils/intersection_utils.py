import numpy as np


def cal_dis(x, y):
    # calculate distance matrix for [x1,xm] and [y1,yn]
    row_x, col_x = x.shape
    row_y, col_y = y.shape
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (row_x, 1)), repeats=row_y, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (row_y, 1)), repeats=row_x, axis=1).T
    dis = x2 + y2 - 2 * xy
    return dis


def find_crossing_merging_split_point(xy1, xy2, dis, th=0.6):
    # minimum distance of each row
    min_dis = dis.min(axis=1)
    # id of y for minimum distance of each row
    id2 = np.argmin(dis, axis=1)
    i = 10
    interaction_points = []
    cnt = 0
    while i < len(min_dis) - 10:
        if min_dis[i - 10] > 0.8*th and min_dis[i + 10] > 0.8*th and\
                0.4*th > min_dis[i]:
            crossing_point = (xy1[i] + xy2[id2[i]]) / 2
            interaction_points.append((crossing_point, i, id2[i], 'crossing_'+str(cnt)))
            cnt += 1
            i += 20
        elif min_dis[i-10] > 0.8*th and min_dis[i] < 0.5 * th \
                and 0.5*th > min_dis[i+5] and 0.5*th > min_dis[i+10]:
            merging_point = (xy1[i]+xy2[id2[i]])/2
            interaction_points.append((merging_point, i, id2[i], 'merging_'+str(cnt)))
            cnt += 1
            i += 20
        elif min_dis[i - 6] < th < min_dis[i + 5] and min_dis[i - 3] < th < 0.8 * min_dis[i + 10] \
                and min_dis[i] < th:
            split_point = (xy1[i] + xy2[id2[i]]) / 2
            interaction_points.append((split_point, i, id2[i], 'split_'+str(cnt)))
            cnt += 1
            i += 20
        i += 1
    return interaction_points


def find_interaction(xy1, xy2, dis_th=2):
    dis = cal_dis(xy1, xy2)
    min_dis = dis.min()
    if min_dis > dis_th:
        # no intersection point
        return None, None
    interaction12 = find_crossing_merging_split_point(xy1, xy2, dis)
    interaction21 = [(m[0], m[2], m[1], m[3]) for m in interaction12]
    return interaction12, interaction21
