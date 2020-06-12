import scipy.io as scio


dataFile = 'D:/Dev/UCB task/Segmented_reference_path_DR_USA_Roundabout_FT.mat'
data = scio.loadmat(dataFile)
print(type(data))
