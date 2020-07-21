#!/usr/bin/env python

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt

import xml.etree.ElementTree as xml
import pyproj
import math

from utils import dict_utils


class Point:
    def __init__(self):
        self.x = None
        self.y = None


class LL2XYProjector:
    def __init__(self, lat_origin, lon_origin):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = math.floor((lon_origin+180.)/6)+1  # works for most tiles, and for all in the dataset
        self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)

    def latlon2xy(self, lat, lon):
        [x, y] = self.p(lon, lat)
        return [x-self.x_origin, y-self.y_origin]


def get_type(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "type":
            return tag.get("v")
    return None


def get_subtype(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "subtype":
            return tag.get("v")
    return None


def get_x_y_lists(element, point_dict):
    x_list = list()
    y_list = list()
    for nd in element.findall("nd"):
        pt_id = int(nd.get("ref"))
        point = point_dict[pt_id]
        x_list.append(point.x)
        y_list.append(point.y)
    return x_list, y_list


def set_visible_area(point_dict, axes):
    min_x = 10e9
    min_y = 10e9
    max_x = -10e9
    max_y = -10e9

    for id, point in dict_utils.get_item_iterator(point_dict):
        min_x = min(point.x, min_x)
        min_y = min(point.y, min_y)
        max_x = max(point.x, max_x)
        max_y = max(point.y, max_y)

    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([min_x - 10, max_x + 10])
    axes.set_ylim([min_y - 10, max_y + 10])
    # print('min_x:', min_x, 'max_x:', max_x)
    # print('min_y:', min_y, 'max_y:', max_y)
    return min_x - 10, max_x + 10, min_y - 10, max_y + 10


def dis(p1, p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5


def link_road(points):
    l1, l2 = -1, -2
    while l1 != l2:
        l1 = len(points)
        i = 0
        while i < len(points):
            j = i+1
            while j < len(points):
                if dis(points[i][0], points[j][-1]) < 2:
                    points[i] = points[j]+points[i]
                    points.pop(j)
                elif dis(points[i][-1], points[j][0]) < 2:
                    points[i] += points[j]
                    points.pop(j)
                elif dis(points[i][0], points[j][0]) < 2:
                    points[i] = points[j][::-1]+points[i]
                    points.pop(j)
                elif dis(points[i][-1], points[j][-1]) < 2:
                    points[i] += points[j][::-1]
                    points.pop(j)
                else:
                    j += 1
            i += 1
        l2 = len(points)
    return points


def draw_map_without_lanelet(filename, axes, lat_origin, lon_origin, scene=''):

    assert isinstance(axes, matplotlib.axes.Axes)

    axes.set_aspect('equal', adjustable='box')
    axes.patch.set_facecolor('white')

    projector = LL2XYProjector(lat_origin, lon_origin)

    e = xml.parse(filename).getroot()

    point_dict = dict()
    for node in e.findall("node"):
        point = Point()
        point.x, point.y = projector.latlon2xy(float(node.get('lat')), float(node.get('lon')))
        point_dict[int(node.get('id'))] = point

    min_x, max_x, min_y, max_y = set_visible_area(point_dict, axes)

    unknown_linestring_types = list()
    road_points = []
    for way in e.findall('way'):
        way_type = get_type(way)
        if way_type is None:
            raise RuntimeError("Linestring type must be specified")
        elif way_type == "curbstone":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif way_type == "line_thin":
            way_subtype = get_subtype(way)
            if way_subtype == "dashed":
                type_dict = dict(color="blue", linewidth=1, zorder=10, dashes=[10, 10])
            else:
                type_dict = dict(color="blue", linewidth=1, zorder=10)
        elif way_type == "line_thick":
            way_subtype = get_subtype(way)
            if way_subtype == "dashed":
                type_dict = dict(color="blue", linewidth=2, zorder=10, dashes=[10, 10])
            else:
                type_dict = dict(color="blue", linewidth=2, zorder=10)
        elif way_type == "pedestrian_marking":
            type_dict = dict(color="red", linewidth=1, zorder=10)
        elif way_type == "bike_marking":
            type_dict = dict(color="blue", linewidth=1, zorder=10, dashes=[5, 10])
        elif way_type == "stop_line":
            type_dict = dict(color="red", linewidth=3, zorder=10)
        elif way_type == "virtual":
            pass
            type_dict = dict(color="blue", linewidth=1, zorder=10, dashes=[2, 5])
        elif way_type == "road_border":
            pass
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif way_type == "guard_rail":
            pass
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif way_type == "traffic_sign":
            continue
        else:
            if way_type not in unknown_linestring_types:
                unknown_linestring_types.append(way_type)
            continue

        x_list, y_list = get_x_y_lists(way, point_dict)
        if way_type not in ["virtual", "road_border"]:
            plt.plot(x_list, y_list, **type_dict)

        if way_type == 'curbstone':
            xy = [(x, y) for x, y in zip(x_list, y_list)]
            flag = 0
            for i, xy0 in enumerate(road_points):
                if dis(xy[0], xy0[-1]) < 2:
                    road_points[i] += xy
                    flag = 1
                    break
                elif dis(xy0[0], xy[-1]) < 2:
                    road_points[i] = xy + xy0
                    flag = 1
                    break
                elif dis(xy[0], xy0[0]) < 2:
                    road_points[i] = xy[::-1] + xy0
                    flag = 1
                    break
                elif dis(xy[-1], xy0[-1]) < 2:
                    road_points[i] = xy0 + xy[::-1]
                    flag = 1
                    break
            if flag == 0:
                road_points.append(xy)
    road_points = link_road(road_points)
    mdis = 25
    for i, road in enumerate(road_points):
        # print(road[0][0], min_x, max_x)
        plt.scatter(road[0][0], road[0][1], zorder=30)
        plt.scatter(road[-1][0], road[-1][1], zorder=30, c='r', alpha=0.5)
        pre = None
        post = None
        corner_x = None
        corner_y = None
        if dis(road[0], road[-1]) < 5:
            continue
        left1 = road[0][0] - min_x
        right1 = max_x - road[0][0]
        top1 = max_y - road[0][1]
        bottom1 = road[0][1] - min_y
        min_dis1 = min([left1, right1, top1, bottom1])

        left2 = road[-1][0] - min_x
        right2 = max_x - road[-1][0]
        top2 = max_y - road[-1][1]
        bottom2 = road[-1][1] - min_y
        min_dis2 = min([left2, right2, top2, bottom2])
        if abs((road[1][1] - road[0][1]) / (road[1][0] - road[0][0])) > 10:
            if road[0][1] > road[1][1]:
                corner_y = max_y
                if top1 < mdis:
                    x = (road[1][0] - road[0][0]) / (road[1][1] - road[0][1]) * (corner_y - road[1][1]) + road[1][0]
                    pre = [[x, max_y]]
                else:
                    pre = [[road[0][0], max_y]]
            else:
                corner_y = min_y
                if bottom1 < mdis:
                    x = (road[1][0] - road[0][0]) / (road[1][1] - road[0][1]) * (corner_y - road[1][1]) + road[1][0]
                    pre = [[x, min_y]]
                else:
                    pre = [[road[0][0], min_y]]
        else:
            if left1 == min_dis1:
                corner_x = min_x
                if left1 < mdis:
                    y = (road[1][1]-road[0][1])/(road[1][0]-road[0][0])*(corner_x-road[1][0])+road[1][1]
                    pre = [[min_x, y]]
                else:
                    pre = [[min_x, road[0][1]]]

            elif right1 == min_dis1:
                corner_x = max_x
                if right1 < mdis:
                    y = (road[1][1] - road[0][1]) / (road[1][0] - road[0][0]) * (corner_x - road[1][0]) + road[1][1]
                    pre = [[max_x, y]]
                else:
                    pre = [[max_x, road[0][1]]]

            elif bottom1 == min_dis1:
                corner_y = min_y
                if bottom1 < mdis:
                    x = (road[1][0] - road[0][0])/(road[1][1] - road[0][1])*(corner_y-road[1][1])+road[1][0]
                    pre = [[x, min_y]]
                else:
                    pre = [[road[0][0], min_y]]

            elif top1 == min_dis1:
                corner_y = max_y
                if top1 < mdis:
                    x = (road[1][0] - road[0][0]) / (road[1][1] - road[0][1]) * (corner_y - road[1][1]) + road[1][0]
                    pre = [[x, max_y]]
                else:
                    pre = [[road[0][0], max_y]]
        # last
        if abs((road[-2][1] - road[-1][1]) / (road[-2][0] - road[-1][0])) > 10:
            if road[-1][1] > road[-2][1]:
                corner_y = max_y
                if top2 < mdis:
                    x = (road[-2][0] - road[-1][0]) / (road[-2][1] - road[-1][1]) * (corner_y - road[-2][1]) + road[-2][
                        0]
                    post = [[x, max_y]]
                else:
                    post = [[road[-1][0], max_y]]
            else:
                corner_y = min_y
                if bottom2 < mdis:
                    x = (road[-2][0] - road[-1][0]) / (road[-2][1] - road[-1][1]) * (corner_y - road[-2][1]) + road[-2][
                        0]
                    post = [[x, min_y]]
                else:
                    post = [[road[-1][0], min_y]]
        else:
            if left2 == min_dis2:
                corner_x = min_x
                if left2 < mdis:
                    y = (road[-2][1] - road[-1][1]) / (road[-2][0] - road[-1][0]) * (corner_x - road[-2][0]) + road[-2][1]
                    post = [[min_x, y]]
                else:
                    post = [[min_x, road[-1][1]]]
            elif right2 == min_dis2:
                corner_x = max_x
                if right2 < mdis:
                    y = (road[-2][1] - road[-1][1]) / (road[-2][0] - road[-1][0]) * (corner_x - road[-2][0]) + road[-2][1]
                    post = [[max_x, y]]
                else:
                    post = [[max_x, road[-1][1]]]

            elif bottom2 == min_dis2:
                corner_y = min_y
                if bottom2 < mdis:
                    x = (road[-2][0] - road[-1][0]) / (road[-2][1] - road[-1][1]) * (corner_y - road[-2][1]) + road[-2][0]
                    post = [[x, min_y]]
                else:
                    post = [[road[-1][0], min_y]]

            elif top2 == min_dis2:
                corner_y = max_y
                if top2 < mdis:
                    x = (road[-2][0] - road[-1][0]) / (road[-2][1] - road[-1][1]) * (corner_y - road[-2][1]) + road[-2][0]
                    post = [[x, max_y]]
                else:
                    post = [[road[-1][0], max_y]]

        if pre:
            road_points[i] = pre + road_points[i]
        if post:
            road_points[i] = road_points[i] + post
        if pre and post:
            road_points[i] += [[corner_x, corner_y]]
    for xy in road_points:
        area = matplotlib.patches.Polygon(xy, closed=True, color='lightgrey')
        axes.add_patch(area)
    if len(unknown_linestring_types) != 0:
        print("Found the following unknown types, did not plot them: " + str(unknown_linestring_types))
