import pickle
import math
import matplotlib.pyplot as plt


work_dir = 'D:/Dev/UCB task/'


def get_thetas(scene='FT'):
    ref_path_info_path = work_dir + 'pickle/{}/ref_path_xy_{}.pkl'.format(scene, scene)
    pickle_file = open(ref_path_info_path, 'rb')
    ref_paths = pickle.load(pickle_file)
    pickle_file.close()
    thetas = []
    for k, v in ref_paths.items():
        print(k)
        for i in range(1, len(v)-1):
            pre = v[i] - v[i-1]
            post = v[i+1] - v[i]
            theta1 = math.atan2(pre[1], pre[0])
            theta2 = math.atan2(post[1], pre[0])
            theta = ((theta2 - theta1)*180/math.pi)
            if theta > 180:
                theta -= 360
            elif theta < -180:
                theta += 360
            thetas.append(theta)
    return thetas


plt.subplot(1, 3, 1)
thetas = get_thetas('FT')
plt.hist(thetas, bins=50)
plt.title('FT')
plt.subplot(1, 3, 2)
thetas = get_thetas('SR')
plt.hist(thetas, bins=50)
plt.title('SR')
plt.subplot(1, 3, 3)
thetas = get_thetas('MA')
plt.hist(thetas, bins=50)
plt.title('MA')
plt.show()
