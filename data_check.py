import random
import pickle
from utils.coordinate_transform import get_xy
from PIL import Image
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches


scene = 'SR'
work_dir = 'D:/Dev/UCB task/'
with open(work_dir + 'pickle/{}/ref_path_xy_{}.pkl'.format(scene, scene), 'rb') as f:
    ref_path_way_points = pickle.load(f)

with open(work_dir + 'pickle/{}/ref_path_frenet_{}.pkl'.format(scene, scene), 'rb') as f:
    ref_frenet = pickle.load(f)

with open(work_dir + 'pickle/{}/track_path_frenet_{}.pkl'.format(scene, scene), 'rb') as f:
    csv_data = pickle.load(f)

with open(work_dir + 'pickle/{}/ts_theta_{}.pkl'.format(scene, scene), 'rb') as f:
    ts_theta = pickle.load(f)

r_key = random.choice(list(csv_data.keys()))

with open(work_dir + 'pickle/{}/edges_{}_{}.pkl'.format(scene, scene, r_key), 'rb') as f:
    edges = pickle.load(f)

with open(work_dir + 'pickle/{}/interaction_{}.pkl'.format(scene, scene), 'rb') as f:
    interactions = pickle.load(f)

r_agent = random.choice(list(csv_data[r_key].keys()))
while r_agent not in edges.keys():
    r_agent = random.choice(list(csv_data[r_key].keys()))

start_ts = csv_data[r_key][r_agent]['time_stamp_ms_first']
end_ts = csv_data[r_key][r_agent]['time_stamp_ms_last']
r_ts = random.choice(range(start_ts, end_ts+100, 100))
r_ms = csv_data[r_key][r_agent]['motion_states'][r_ts]
ref_path_name = csv_data[r_key][r_agent]['ref path']
print('Random sample: ', scene, r_key, r_agent, r_ts, ref_path_name)
x = r_ms['x']
y = r_ms['y']
s = r_ms['frenet_s']
d = r_ms['frenet_d']
trans_x, trans_y = get_xy(s, d, ref_frenet[ref_path_name], ref_path_way_points[ref_path_name])
print('x,y: ', x, y, 'transformed x,y: ', trans_x, trans_y)

single_path_img = 'D:/Dev/UCB task/intersection_figs/single_{}/{}.png'.format(scene, ref_path_name)
x, y, psi_rad = ts_theta[r_key][r_agent][r_ts]
theta = math.pi / 2 - psi_rad
img = Image.open(single_path_img)
w, h = img.size
start_xy = {'SR': (900, 965), 'FT': (945, 945), 'MA': (955, 945)}
xs, ys = start_xy[scene]
xc = int(1 + (x - xs) * 1.6)
yc = h - int(1 + (y - ys) * 1.6)
img = img.rotate(theta * 180 / math.pi, center=(xc, yc), resample=Image.BILINEAR)
img = img.crop((xc - 16, yc - 32, xc + 16, yc))

# fig, axes = plt.subplots(1, 1, figsize=(30, 20), dpi=100)
# plt.imshow(img)
# circle = patches.Circle((16, 32), 1, color='r', zorder=3)
# axes.add_patch(circle)
# plt.show()


r_start_ts = random.choice(list(edges[r_agent].keys()))
r_id1 = random.choice(list(edges[r_agent][r_start_ts]['ita_id'].keys()))
path1 = csv_data[r_key][r_id1]['ref path']
r_id2 = random.choice(list(edges[r_agent][r_start_ts]['ita_id'][r_id1].keys()))
ita_id = edges[r_agent][r_start_ts]['ita_id'][r_id1][r_id2]
path2 = csv_data[r_key][r_id2]['ref path']
while path1 == path2:
    r_start_ts = random.choice(list(edges[r_agent].keys()))
    r_id1 = random.choice(list(edges[r_agent][r_start_ts]['ita_id'].keys()))
    path1 = csv_data[r_key][r_id1]['ref path']
    r_id2 = random.choice(list(edges[r_agent][r_start_ts]['ita_id'][r_id1].keys()))
    ita_id = edges[r_agent][r_start_ts]['ita_id'][r_id1][r_id2]
    path2 = csv_data[r_key][r_id2]['ref path']

pair = sorted([path1, path2])
ita_point, seq_id1, seq_id2, ita_type = interactions[pair[0]][pair[1]][ita_id]
print('ita type:', ita_type)

r_20_ts = r_start_ts + 20 * 100
ms1 = csv_data[r_key][r_id1]['motion_states'][r_20_ts]
x1 = ms1['x']
y1 = ms1['y']
x1 = x1 - ita_point[0] + 20
y1 = ita_point[1] - y1 + 20
x1_seq = []
y1_seq = []
for i in range(10):
    x1_seq.append(csv_data[r_key][r_id1]['motion_states'][r_20_ts+i*100]['x'] - ita_point[0] + 20)
    y1_seq.append(ita_point[1] - csv_data[r_key][r_id1]['motion_states'][r_20_ts+i*100]['y'] + 20)
ms2 = csv_data[r_key][r_id2]['motion_states'][r_20_ts]
x2 = ms2['x']
y2 = ms2['y']
x2 = x2 - ita_point[0] + 20
y2 = ita_point[1] - y2 + 20
x2_seq = []
y2_seq = []
for i in range(10):
    x2_seq.append(csv_data[r_key][r_id2]['motion_states'][r_20_ts+i*100]['x'] - ita_point[0] + 20)
    y2_seq.append(ita_point[1] - csv_data[r_key][r_id2]['motion_states'][r_20_ts+i*100]['y'] + 20)

psi1 = ms1['psi_rad']
theta1 = math.pi / 2 - psi1
print(psi1)
ita_img = Image.open(work_dir + 'intersection_figs/roundabout_{}_crop/{}_{}_{}_0.png'.format(scene, pair[0], pair[1], ita_id))
ita_img_r = ita_img.rotate(theta1*180/math.pi)
ita_img_rc = ita_img_r.crop((8, 8, 40, 40))
plt.subplot(121)
plt.imshow(ita_img)
plt.title('raw img')
plt.scatter(x1_seq, y1_seq, c='r', s=4, marker='x')
plt.quiver(x1, y1, 3*math.cos(psi1), 3*math.sin(psi1), color='r', width=0.005)
plt.scatter(x2_seq, y2_seq, c='g', s=4, marker='x')
plt.quiver(x2, y2, 3*math.cos(ms2['psi_rad']), 3*math.sin(ms2['psi_rad']), color='g', width=0.005)
plt.subplot(122)
plt.title('rotate crop 32*32')
plt.imshow(ita_img_rc)
plt.show()
