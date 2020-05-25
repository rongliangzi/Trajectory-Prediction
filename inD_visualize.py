from PIL import Image
import pylab
import csv
import numpy as np

if __name__ == '__main__':
    data_dir = 'D:/Dev/UCB task/inD-dataset-v1.0/data/'
    fig, axes = pylab.subplots(1, 1, figsize=(12, 8), dpi=250)
    img = pylab.array(Image.open(data_dir+'00_background.png'))
    pylab.imshow(img)

    poly_file = 'D:/Dev/UCB task/inD/ref_path_scenario1.csv'
    with open(poly_file, 'r') as f:
        reader = csv.reader(f)
        for row_id, row_data in enumerate(reader):
            row_id -= 1
            if row_id not in [0, 5]:
                continue
            mode = float(row_data[5])
            coef = []
            sp = row_data[4][1:-1].split(' ')
            for s in sp:
                if s:
                    coef += [float(s)]
            # coef = coef[::-1]
            poly_func = np.poly1d(coef)
            if mode == 0:
                x = np.arange(img.shape[1]*0.7, img.shape[1]*0.8, 1)
                y = poly_func(x)
            else:
                y = np.arange(img.shape[0]*0.4, img.shape[0]*0.8, 1)
                x = poly_func(y)
            # y = img.shape[0]-y
            pylab.plot(x, y, c='r')
    # 添加标题，显示绘制的图像
    pylab.title('Plotting:"pic1.png"')
    pylab.axis('off')
    pylab.show()
