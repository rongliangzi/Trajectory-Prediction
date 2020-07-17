import csv
import glob
import os
r_areas = [str(i) for i in range(11)]
t_areas = [str(i+11) for i in range(7)]
paths = glob.glob(os.path.join('D:/Dev/UCB task/Roundabout_EP_final/track_updated/', '*.csv'))
paths.sort()
for csv_name in paths:
    with open(csv_name) as csv_file:
        t_csv = open('D:/Dev/UCB task/Roundabout_EP_final/track_T/{}.csv'.format(csv_name[-7:-4]), "w", newline='')
        t_writer = csv.writer(t_csv)
        r_csv = open('D:/Dev/UCB task/Roundabout_EP_final/track_R/{}.csv'.format(csv_name[-7:-4]), "w", newline='')
        r_writer = csv.writer(r_csv)
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(list(csv_reader)):
            if i == 0:
                t_writer.writerow(row)
                r_writer.writerow(row)

            if row[11] == 'NAN' or row[12] == 'NAN':
                continue
            elif row[11] in t_areas and row[12] in t_areas:
                t_writer.writerow(row)
            elif row[11] in r_areas and row[12] in r_areas:
                r_writer.writerow(row)
        t_csv.close()
        r_csv.close()