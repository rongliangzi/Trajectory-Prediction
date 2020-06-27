import os
import glob


paths = glob.glob(os.path.join('D:/Dev/UCB task/intersection_figs/roundabout_FT_crop/', '*_99.png'))
rpaths = glob.glob(os.path.join('D:/Dev/UCB task/intersection_figs/roundabout_FT/', '*.png'))
for path, rp in zip(paths, rpaths):

    img_name = path.split('\\')[-1].split('_99.png')[0]
    if not os.path.exists('D:/Dev/UCB task/intersection_figs/roundabout_FT/'+img_name+'.png'):
        print(img_name)
