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


def find_crossing_point(dis, x, y, th=0.2):
    # minimum distance of each row
    min_dis = dis.min(axis=1)
    # id of y for minimum distance of each row
    y_id = np.argmin(dis, axis=1)
    assert len(min_dis) > 10
    # try to find a crossing point.
    # two points before it > th and two points behind it > th
    # if not found, return None
    i = 6
    crossing_points = []
    cnt = 0
    while i < len(min_dis) - 6:
        if min_dis[i - 5] > 2*th and min_dis[i - 6] > 2*th and\
                min_dis[i + 5] > 2*th and min_dis[i + 6] > 2*th and\
                min_dis[i] < th:
            crossing_point = (x[i] + y[y_id[i]]) / 2
            crossing_points.append((crossing_point, i, y_id[i], 'crossing_'+str(cnt)))
            cnt += 1
            i += 50
        i += 1
    return crossing_points


def find_merging_point(x, y, dis, th=0.8):
    # minimum distance of each row
    min_dis = dis.min(axis=1)
    # id of y for minimum distance of each row
    y_id = np.argmin(dis, axis=1)
    assert len(min_dis) > 5
    # try to find a merging point. two points before it > th and two points behind it < th
    # if not found, return None
    i = 10
    merging_points = []
    cnt = 0
    while i < len(min_dis)-10:
        if min_dis[i-10] > 2*th > 2*min_dis[i+5] and min_dis[i-5] > 1.5*th > 1.5*min_dis[i+10]\
                and min_dis[i] < th:
            merging_point = (x[i]+y[y_id[i]])/2
            merging_points.append((merging_point, i, y_id[i], 'merging_'+str(cnt)))
            cnt += 1
            i += 50
        i += 1
    return merging_points


def find_split_point(xy1, xy2, th=1):
    dis = cal_dis(xy1, xy2)
    # minimum distance of each row
    min_dis = dis.min(axis=1)
    # id of y for minimum distance of each row
    id2 = np.argmin(dis, axis=1)
    i = 6
    cnt = 0
    while i < len(min_dis) - 10:
        if min_dis[i - 6] < th < min_dis[i + 5] and min_dis[i - 3] < th < 0.8 * min_dis[i + 10] \
                and min_dis[i] < th:
            split_point = (xy1[i] + xy2[id2[i]]) / 2
            return split_point, i, id2[i], 'split_' + str(cnt)
        i += 1
    return None


def find_intersection(point_x1, point_y1, point_x2, point_y2, dis_th=2, mg_th=0.8):
    x = np.array([[x1, y1] for x1, y1 in zip(point_x1, point_y1)])
    y = np.array([[x2, y2] for x2, y2 in zip(point_x2, point_y2)])
    dis = cal_dis(x, y)
    min_dis = dis.min()

    if min_dis > dis_th:
        # no intersection point
        return None
    merging = find_merging_point(x, y, dis, mg_th)
    intersection = []
    if len(merging) > 0:
        intersection += merging
    crossing = find_crossing_point(dis, x, y)
    if len(crossing) > 0:
        intersection += crossing
    return intersection
