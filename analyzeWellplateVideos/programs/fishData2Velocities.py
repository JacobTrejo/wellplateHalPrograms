import numpy as np
from scipy import ndimage
from tqdm import tqdm
from scipy.signal import savgol_filter

def fishData2Velocity(fishData, ccData, threshold = .7, fps = 50):

    for fishIdx in range(fishData.shape[1]):
        ccs = ccData[:, fishIdx]
        fishData[ccs < threshold, fishIdx, ...] = np.nan


    # Lets turn into the poses which fall below the threshold into nan

    amountOfFrames, amountOfFish = fishData.shape[:2]
    
    velocityData = np.zeros((amountOfFrames - 1, amountOfFish, 5))

    # Lets smooth out the data

    sigma = 2

    fishPoseList = []

    for fishIdx in range(fishData.shape[1]):
        fish = fishData[:, fishIdx, ...]
        smoothXs = []
        smoothYs = []

        for ptIdx in range(12):
            x = fish[:,0,ptIdx]
            y = fish[:,1,ptIdx]

            #smoothFish = ndimage.gaussian_filter1d(firstFish[...,ptIdx], sigma, axis = 1)
            #xSmooth = smoothFish[:,0]
            #ySmooth = smoothFish[:,1]

            ## Smoothing with gaussian filter
            #xSmooth = ndimage.gaussian_filter1d(x, sigma)
            #ySmooth = ndimage.gaussian_filter1d(y, sigma)

            ## Smoothing with Savitsky-Golay filter
            window = 21
            degree = 3
            xSmooth = savgol_filter(x, window, degree)
            ySmooth = savgol_filter(y, window, degree)

            smoothXs.append(xSmooth)
            smoothYs.append(ySmooth)
        fishPoseList.append((smoothXs, smoothYs))

    COMVs = []
    maxLenght = 0
    for fishIdx in range(fishData.shape[1]):
        smoothXs, smoothYs = fishPoseList[fishIdx]

        COMX = smoothXs[2]
        COMY = smoothYs[2]

        DT = (1/fps)
        DX = COMX[1:] - COMX[:-1]
        DY = COMY[1:] - COMY[:-1]

        VX = DX / DT
        VY = DY / DT

        length = (VX ** 2 + VY ** 2 )**.5
        VXN = VX / length
        VYN = VY / length
        
        velocityData[:, fishIdx, 0] = VXN
        velocityData[:, fishIdx, 1] = COMX[:-1]
        velocityData[:, fishIdx, 2] = VYN
        velocityData[:, fishIdx, 3] = COMY[:-1]
        velocityData[:, fishIdx, 4] = length

        COMVs.append((VXN, VYN, length))

        maxLenghtInstance = np.max(length)
        if maxLenghtInstance > maxLenght: maxLenght = maxLenghtInstance

    return velocityData










