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


def find_crossing_merging_split_point(xy1, xy2, dis, th, skip, k=1):
    # minimum distance of each row
    min_dis = dis.min(axis=1)
    # id of y for minimum distance of each row
    id2 = np.argmin(dis, axis=1)
    i = 10*k
    interaction_points = []
    cnt = 0
    while i < len(min_dis) - 10*k:
        if min_dis[i - 10*k] > 0.8*th and min_dis[i + 10*k] > 0.8*th and\
                0.4*th > min_dis[i]:
            crossing_point = (xy1[i] + xy2[id2[i]]) / 2
            interaction_points.append((crossing_point, i, id2[i], 'crossing_'+str(cnt)))
            cnt += 1
            i += skip
        elif min_dis[i-10*k] > 0.8*th and min_dis[i] < 0.5 * th \
                and 0.5*th > min_dis[i+5*k] and 0.5*th > min_dis[i+10*k]:

            merging_point = (xy1[i]+xy2[id2[i]])/2
            interaction_points.append((merging_point, i, id2[i], 'merging_'+str(cnt)))
            cnt += 1
            i += skip
        elif min_dis[i - 6*k] < th < min_dis[i + 5*k] and min_dis[i - 3*k] < th < 0.8 * min_dis[i + 10*k] \
                and min_dis[i] < th:
            split_point = (xy1[i] + xy2[id2[i]]) / 2
            interaction_points.append((split_point, i, id2[i], 'split_'+str(cnt)))
            cnt += 1
            i += skip
        i += 1
    return interaction_points


# used in FT/SR dataset
def find_interaction(xy1, xy2, th, skip, k=1, dis_th=2,):
    dis = cal_dis(xy1, xy2)
    min_dis = dis.min()
    if min_dis > dis_th:
        # no intersection point
        return None, None
    interaction12 = find_crossing_merging_split_point(xy1, xy2, dis, th, skip, k)
    interaction21 = [(m[0], m[2], m[1], m[3]) for m in interaction12]
    return interaction12, interaction21


# used in MA dataset
def find_intersection_ita(path1, path2, xy1, xy2, th, skip, dis_th=2, m=1.5, k=-1, insert_k=1):
    dis = cal_dis(xy1, xy2)
    min_dis = dis.min()
    if min_dis > dis_th:
        # no intersection point
        return None, None
    if path1.split('-')[0] == path2.split('-')[0]:
        interaction12 = find_intersection_split(xy1, xy2, dis, th, skip, k, insert_k)
    elif path1.split('-')[-1] == path2.split('-')[-1]:
        interaction12 = find_intersection_merging(xy1, xy2, dis, th, skip, m, k, insert_k)
    else:
        interaction12 = find_intersection_crossing(xy1, xy2, dis, th, skip, k, insert_k)
    interaction21 = [(m[0], m[2], m[1], m[3]) for m in interaction12]
    return interaction12, interaction21


def find_intersection_crossing(xy1, xy2, dis, th, skip, k=-1, insert_k=1):
    skip = skip * insert_k
    # minimum distance of each row
    min_dis = dis.min(axis=1)
    # id of y for minimum distance of each row
    id2 = np.argmin(dis, axis=1)
    if k == -1:
        k = 20
    i = k*insert_k
    crossing_points = []
    cnt = 0
    while i < len(min_dis) - k*insert_k:
        if min_dis[i - k*insert_k] > 0.6*th and min_dis[i + k*insert_k] > 0.6*th and\
                0.4*th > min_dis[i]:
            crossing_point = (xy1[i] + xy2[id2[i]]) / 2
            crossing_points.append((crossing_point, i, id2[i], 'crossing_'+str(cnt)))
            cnt += 1
            i += skip
            return crossing_points
        i += 1
    return crossing_points


def find_intersection_merging(xy1, xy2, dis, th, skip, m=1.5, k=-1, insert_k=1):
    skip = skip * insert_k
    # minimum distance of each row
    min_dis = dis.min(axis=1)
    # id of y for minimum distance of each row
    id2 = np.argmin(dis, axis=1)
    if k == -1:
        k = 10
    i = k * insert_k
    merging_points = []
    cnt = 0
    while i < len(min_dis) - k*insert_k:
        if min_dis[i - k*insert_k] > m*0.4 * th and min_dis[i] < 0.4 * th \
                and 0.4 * th > min_dis[i + k*insert_k]:
            if id2[i-k*insert_k] == id2[i-k*insert_k//2] or id2[i+k*insert_k//2] == id2[i+k*insert_k]:
                pass
            else:
                merging_point = (xy1[i] + xy2[id2[i]]) / 2
                merging_points.append((merging_point, i, id2[i], 'merging_' + str(cnt)))
                cnt += 1
                i += skip
                return merging_points
        i += 1
    return merging_points


def find_intersection_split(xy1, xy2, dis, th, skip, k=-1, insert_k=1):
    # minimum distance of each row
    skip = skip * insert_k
    min_dis = dis.min(axis=1)
    # id of y for minimum distance of each row
    id2 = np.argmin(dis, axis=1)
    if k == -1:
        k = 5
    i = k*insert_k
    split_points = []
    cnt = 0
    while i < len(min_dis)-2*k*insert_k:
        if min_dis[i] < th < min_dis[i + k*insert_k] and min_dis[i - 3*insert_k] < th < 0.8 * min_dis[i + 2*k*insert_k]:
            if id2[i-k*insert_k] == id2[i-k*2*insert_k] or id2[i+k*2*insert_k] == id2[i+k*insert_k]:
                pass
            else:
                split_point = (xy1[i] + xy2[id2[i]]) / 2
                split_points.append((split_point, i, id2[i], 'split_' + str(cnt)))
                cnt += 1
                i += skip
                return split_points
        i += 1
    return split_points
