from utils import map_vis_without_lanelet
import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np
fig, axes = plt.subplots(1, 1)
rect = matplotlib.patches.Polygon([(0,10), (10,8), (10, 0)], closed=True)
axes.add_patch(rect)
# rect = matplotlib.patches.Polygon([(2,10), (8,8), (8, 0), (2,0)], closed=True, color='white')
# axes.add_patch(rect)
plt.xlim(-20,20)
plt.ylim(-20,20)

plt.show()