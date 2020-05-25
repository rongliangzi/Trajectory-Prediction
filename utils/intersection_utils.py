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


def find_crossing_point(dis, x, y):
    pos = np.argmin(dis)
    xid = pos // y.shape[0]
    yid = pos % y.shape[0]
    # the shortest distance is from xid-th point in x to yid-th point in y
    crossing_point = (x[xid]+y[yid])/2
    # print('({:.3f},{:.3f})'.format(crossing_point[0], crossing_point[1]))
    return crossing_point, xid, yid


def find_merging_point(x, y, dis, th=0.8):
    # minimum distance of each row
    min_dis = dis.min(axis=1)
    # id of y for minimum distance of each row
    y_id = np.argmin(dis, axis=1)
    assert len(min_dis) > 5
    # try to find a merging point. two points before it > th and two points behind it < th
    # if not found, return None
    for i in range(2, len(min_dis)-2):
        if min_dis[i-1] > th > min_dis[i+1] and min_dis[i-2] > th > min_dis[i+2]:
            merging_point = (x[i]+y[y_id[i]])/2
            return merging_point, i, y_id[i]
        return None


def find_intersection(seq1, seq2):
    X1, Y1 = seq1[0], seq1[1]
    X2, Y2 = seq2[0], seq2[1]
    x = np.array([[x1, y1] for x1, y1 in zip(X1, Y1)])
    y = np.array([[x2, y2] for x2, y2 in zip(X2, Y2)])
    dis = cal_dis(x, y)
    min_dis = dis.min()

    if min_dis > 15:
        # no intersection point
        return None
    intersection = find_merging_point(x, y, dis)
    # no merging point, find crossing point
    if intersection is None:
        intersection, xid, yid = find_crossing_point(dis, x, y)
    else:
        intersection, xid, yid = intersection
    return intersection, xid, yid
