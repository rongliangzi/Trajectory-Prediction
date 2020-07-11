def rotate_crop_ts(img_path, data, xs, ys):
    import matplotlib.pyplot as plt
    import math
    x, y, psi_rad = data
    theta = math.pi / 2 - psi_rad
    from PIL import Image
    img = Image.open(img_path)
    w, h = img.size
    xc = int(1 + (x - xs) * 1.6)
    yc = h - int(1 + (y - ys) * 1.6)
    img = img.rotate(theta * 180 / math.pi, center=(xc, yc), resample=Image.BILINEAR)
    img = img.crop((xc - 16, yc - 16, xc + 16, yc + 16))

    print(img.size, xc, yc)
    plt.figure()
    plt.imshow(img)
    plt.show()
    return img


if __name__ == '__main__':
    dataset = 'SR'
    import pickle

    # path
    pickle_file = open(your_path+'ts_theta_{}.pkl'.format(dataset), 'rb')
    data = pickle.load(pickle_file)
    pickle_file.close()
    # pre defined border of x and y, for coordinate transformation
    start_xy = {'SR': (900, 965), 'FT': (945, 945), 'MA': (955, 945)}
    xs, ys = start_xy[dataset]
    # in 000.csv, for the car id=11 whose ref path='5--1-2', in time step=fts, get the 32*32 image
    # data[csv_id][car_id][time step] return [x, y, psi_rad]
    path = '5--1-2'
    fts = 3600
    img = rotate_crop_ts(your_path+'single_SR/{}.png'.format(path), data['000'][11][fts], xs, ys)
