import numpy as np
import cv2 as cv
import os
from multiprocessing import  Pool, Manager

#from programs.calculate_intrinsic_parameters import calculate_intrinsic_parameters, calculate_seglen
from programs.calculate_intrinsic_parameters_hc_constant import calculate_intrinsic_parameters, calculate_seglen

from tqdm import tqdm
from scipy.stats import norm

filename = 'IntrinsicParameters.yaml'


uploadFolder = 'uploadFolder/'
folder = os.listdir(uploadFolder)
parentFolder = uploadFolder + folder[0]


#parentFolder = '../cutout_data'
subFolder = ['images', 'points']

imagesFolder = parentFolder + '/' + subFolder[0]
pointsFolder = parentFolder + '/' + subFolder[1]

outputsFolder = 'outputs/'
recreatedOutputsFolder = 'outputs_recreated/'
# preparing the data

imagesList = os.listdir(imagesFolder)
imagesList.sort()
pointsList = os.listdir(pointsFolder)
pointsList.sort()


def calculateAndSaveData(ins):
    idx, xs, seglens = ins
    imPath = imagesFolder + '/' + imagesList[idx]
    pointsPath = pointsFolder + '/' + pointsList[idx]

    im = cv.imread(imPath)
    points = np.load(pointsPath)    
    
    name = pointsPath.split('/')[-1]
    name = name[:-4] + '.png'
    x, seglen, recreatedIm = calculate_intrinsic_parameters(im, points)
    
    xs.append(x)
    seglens.append(seglen)
    #cv.imwrite(recreatedOutputsFolder + name, recreatedIm)

manager = Manager()
xList = manager.list()
seglens = manager.list()

print('calculating the intrinsic parameters')

poolObj = Pool()
#poolObj.map(calculateAndSaveData, range(0, 15))
#poolObj.map(calculateAndSaveData,[ (idx , xList, seglens ) for idx in range( len( imagesList)) ]  )
list(tqdm(poolObj.imap(calculateAndSaveData,[ (idx , xList, seglens ) for idx in range( len( imagesList)) ]  ), total = len(imagesList)))
poolObj.close()

xVectors = np.array([])
xList = list(xList)
for x in xList:
    xVectors = np.vstack((xVectors, x)) if xVectors.size > 0 else x

seglens = list(seglens)
seglens = np.array(seglens)

# Saving the vectors if you wish to analyze them
np.save('xVectors.npy', xVectors)
np.save('seglens.npy', seglens)

# Writting the yaml file
modelValueNames = ['d_eye', 'c_eye', 'c_belly', 'c_head', 'eye_br', 'belly_br', 'head_br', 'eye_w', 'eye_l',
                           'belly_w', 'belly_l', 'head_w', 'head_l', 'ball_size', 'ball_thickness', 'tail_br', 'seglen']
modelValues = []
ip = xVectors[:,3:]

for pIdx in range(ip.shape[1] + 1):
    if pIdx == ip.shape[1]:
        values = seglens
    else:
        values = ip[:, pIdx]

    mu, std = norm.fit(values)
    modelValues.append([mu, std])

# Actually writing the file
ipFile = open(filename, 'w')

for parameterValues, parameterName in zip(modelValues, modelValueNames):
    mu, std = parameterValues

    # Writting the distribution
    distributionLine = parameterName + '_distribution: np.random.normal('
    distributionLine += str(mu) + ' ,' + str(std) + ' )\n'
    ipFile.write(distributionLine)

    # Writting the mean
    meanLine = parameterName + '_u: ' + str(mu) + ' \n'
    ipFile.write(meanLine)

ipFile.close()




