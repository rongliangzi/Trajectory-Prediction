import pickle
import os


if os.path.exists('D:/Dev/UCB task/pickle/track_path_frenet_SR.pkl'):
    pickle_file = open('D:/Dev/UCB task/pickle/track_path_frenet_SR.pkl', 'rb')
    csv_data = pickle.load(pickle_file)
    pickle_file.close()
