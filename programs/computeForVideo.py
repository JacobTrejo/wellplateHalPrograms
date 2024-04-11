
from programs.ResNet_Blocks_3D_four_blocks import resnet18
#from CustomDataset2 import CustomImageDataset
#from CustomDataset2 import padding

#from CustomDataset import CustomImageDataset
#from CustomDataset import padding

from programs.CustomDataset2 import CustomImageDataset
from programs.CustomDataset2 import padding

import torch
import numpy as np
import cv2 as cv
import imageio
import pdb

import torchvision
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from programs.evaluation_functions import evaluate_prediction
from programs.evaluation_functions import interparc, normalize_data  

import time
from multiprocessing import Process, Manager, Pool
import multiprocessing

from tqdm import tqdm

ccThreshold = -2

#bgsub_im = imageio.imread('../well_plates_bgsub.png')
#bgsub_im = np.asarray(bgsub_im)
videoFileName = 'testImagingBsub.npy'
videoName = videoFileName[:-4]

gridFolder = 'uploadYourGridHere/'
firstGridFile = gridFolder +  os.listdir(gridFolder)[0]
grid = np.load(firstGridFile)

amountOfCircles = grid.shape[0]

videoFolder = 'inputs/bgsub_videos/'
videoPath = videoFolder + videoFileName


resnetWeightsFolder = '../weights/resnet/'
resnetWeightsFolder = '../'
resnetWeightsFile = 'resnet_pose_best_python_230608_four_blocks.pt'
resnetWeightsFile = 'resnet_pose_best_python_230608_four_blocks.pt'
resnetWeights = resnetWeightsFolder + resnetWeightsFile
resnetWeights = '../../hardcodedWellsIntrinsic/Resnet/resnet_pose_best_python_230608_four_blocks.pt'
resnetWeights = 'weights/first/resnet_pose_best_python_230608_four_blocks.pt'

resnetWeightFolder = 'inputs/weights/runmeWeight/'
resnetWeights = resnetWeightFolder + os.listdir(resnetWeightFolder)[0]


red = [0,0,255]
green = [0,255,0]
blue = [255, 0, 0]

inputsFolder = 'inputs/'
outputsFolder = 'outputs/'

# resnet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnetModel = resnet18(1, 12, activation='leaky_relu').to(device)
resnetModel = nn.DataParallel(resnetModel)
resnetModel.load_state_dict(torch.load( resnetWeights  ))
resnetModel.eval()

n_cuda = torch.cuda.device_count()
if (torch.cuda.is_available()):
    print(str(n_cuda) + 'GPUs are available!')
    nworkers = n_cuda*12
    pftch_factor = 2
else:
    print('Cuda is not available. Training without GPUs. This might take long')
    nworkers = 4
    pftch_factor = 2
batch_size = 512*n_cuda

#if torch.cuda.device_count() > 1:
#  print("Using " + str(n_cuda) + " GPUs!")
#  model = nn.DataParallel(model)

# Getting the cutouts
#for circ in grid:
#    center = (int(circ[0]),int(circ[1]) )
#    radius = int(circ[2])
#
#    sX = center[0] - radius
#    bX = center[0] + radius
#    sY = center[1] - radius
#    bY = center[1] + radius
#    cutOut = result[sY:bY + 1, sX:bX + 1]

def interplatePts(pt):
    vec = np.diff(pt, n = 1, axis = 1)
    #theta_0 = np.arctan2(vec[1, 0], vec[0, 0])
    #theta_i = np.diff(np.arctan2(vec[1, 0:9], vec[0,0:9]))
    t = normalize_data(np.concatenate((np.array([0, 0.8]),  0.8 + np.cumsum(np.ones((1, 9), dtype = np.float64)))))
    pt_for_interpolation = np.concatenate((np.mean(pt[:, 10 : 12, None], axis = 1), pt[:, 0 : 10]), axis = 1)
    pt = interparc(t, pt_for_interpolation[0, :], pt_for_interpolation[1, :]).T
    return pt

def superImpose(image):
    rgb = np.stack((image, image, image), axis = 2)
    cc_list = []    
    global grid
    # getting the cutOuts
    cutOutList = []
    for circ in grid:
        center = (int(circ[0]),int(circ[1]) )
        radius = int(circ[2])
    
        sX = center[0] - radius
        bX = center[0] + radius
        sY = center[1] - radius
        bY = center[1] + radius
        cutOut = image[sY:bY + 1, sX:bX + 1]
        
        cutOut = cutOut.astype(float)
        cutOut *= 255 / np.max(cutOut)
        cutOut = cutOut.astype(np.uint8)

        cutOutList.append(cutOut)
    
    # Prepping the data to give to resnet
    transform = transforms.Compose([padding(), transforms.PILToTensor() ])
    data = CustomImageDataset(cutOutList, transform=transform)
    loader = DataLoader(data, batch_size=batch_size,shuffle=False,num_workers=nworkers,prefetch_factor=pftch_factor,persistent_workers=True)

    for i, im in enumerate(loader):
        im = im.to(device)
        pose_recon = resnetModel(im)

        #pose_recon = pose_recon.detach().cpu().numpy()
        #im = np.squeeze(im.detach().cpu().numpy())

        pose_recon = pose_recon.detach().cpu().numpy()
        im = np.squeeze(im.cpu().detach().numpy())


        for imIdx in range(im.shape[0]):
            im1 = im[imIdx,...]
            im1 *= 255
            im1 = im1.astype(np.uint8)
            pt1 = pose_recon[imIdx,...]
            
            noFishThreshold = 10
            if np.max(pt1) < noFishThreshold: continue
            
            cc = evaluate_prediction(im1, pt1)
            cc_list.append(cc)
    return cc_list



def getResultsFromResnet(images, batchImageIdx0):
    resultsFromResnet = []
    #rgb = np.stack((image, image, image), axis = 2)
    cc_list = []
    global amountOfCircles 
    global grid
    circleIdx = 0
    imageIdx0Offset = 0

    # getting the cutOuts
    cutOutList = []
    for image in images:
        for circ in grid:
            center = (int(circ[0]),int(circ[1]) )
            radius = int(circ[2])

            sX = center[0] - radius
            bX = center[0] + radius
            sY = center[1] - radius
            bY = center[1] + radius
            
            cutOut = image[sY:bY + 1, sX:bX + 1]

            cutOut = cutOut.astype(float)
            cutOut *= 255 / np.max(cutOut)
            cutOut = cutOut.astype(np.uint8)

            cutOutList.append(cutOut)

    # Prepping the data to give to resnet
    transform = transforms.Compose([padding(), transforms.PILToTensor() ])
    data = CustomImageDataset(cutOutList, transform=transform)
    loader = DataLoader(data, batch_size=batch_size,shuffle=False,num_workers=nworkers,prefetch_factor=pftch_factor,persistent_workers=True)

    for i, im in enumerate(loader):
        im = im.to(device)
        pose_recon = resnetModel(im)

        #pose_recon = pose_recon.detach().cpu().numpy()
        #im = np.squeeze(im.detach().cpu().numpy())

        pose_recon = pose_recon.detach().cpu().numpy()
        im = np.squeeze(im.cpu().detach().numpy(), axis = 1)


        for imIdx in range(im.shape[0]):
            im1 = im[imIdx,...]
            im1 *= 255
            im1 = im1.astype(np.uint8)
            pt1 = pose_recon[imIdx,...]

            noFishThreshold = 10
            
            if np.max(pt1) < noFishThreshold:
                # placeholder
                jj = 5
            else:
                resultsFromResnet.append((im1, pt1,batchImageIdx0 + imageIdx0Offset, circleIdx))
                #cc = evaluate_prediction(im1, pt1)
                #cc_list.append(cc)
                #print(cc)
            circleIdx += 1
            if circleIdx == amountOfCircles:
                circleIdx = 0
                imageIdx0Offset += 1
    return resultsFromResnet

def getCC(ins):
    global grid
    global ccThreshold

    im1, pt1, frameIdx, circleIdx, cc_list = ins
    cc = evaluate_prediction(im1, pt1)
    
    if cc > ccThreshold or ccThreshold < -1:
        # Lets get the points from the reference frame of the whole image
        nonZero = np.where( im1 > 0  )
        sY = np.min( nonZero[0] )
        sX = np.min( nonZero[1] )
        pt1[0,:] -= sX
        pt1[1,:] -= sY

        circ = grid[circleIdx]
        center = (int(circ[0]),int(circ[1]) )
        radius = int(circ[2])

        sX = center[0] - radius
        bX = center[0] + radius
        sY = center[1] - radius
        bY = center[1] + radius
        #sX, sY, bX, bY = boxes[ imIdx, ...]
        pt1[0,:] += sX
        pt1[1,:] += sY

        #interPts = interplatePts(pt1)
        #print('The shape of pt: ', pt1.shape)
        #pt1[:,:10] = interPts[:,:10]
        
        cc_list.append( (frameIdx, circleIdx, pt1, cc))
        #print(cc)
    
    return   

# Initializing video capture object and getting parameters
#vidcap = cv.VideoCapture(videoPath)
#fps = vidcap.get( cv.CAP_PROP_FPS )
#height = vidcap.get( cv.CAP_PROP_FRAME_HEIGHT )
#width = vidcap.get( cv.CAP_PROP_FRAME_WIDTH )


def getCorrelationCoefficientsFromVideo(vid): 
    
    #vid = vid[:100, ...]
    global grid
    global amountOfCircles
    gridFolder = 'uploadYourGridHere/'
    firstGridFile = gridFolder +  os.listdir(gridFolder)[0]
    grid = np.load(firstGridFile)

    #vid = np.load(videoPath)
    amountOfFrames = vid.shape[0]
    
    pbar = tqdm(total = amountOfFrames)
    #amountOfFrames = 50
    
    # This will be the array to keep the pose data of the fish
    fishData = np.zeros((amountOfFrames, amountOfCircles, 2, 12))
    fishData.fill(np.nan)
    ccData = np.zeros((amountOfFrames, amountOfCircles))
    ccData.fill(np.nan)

    fps = 100
    height, width = vid.shape[1:3]
    grid *= width
    # NOTE: currently having a well of a greater size that 101 will cause an offset
    grid[:,2] = np.clip(grid[:,2], 0, 49)

    # Getting the output ready
    #fourcc = cv.VideoWriter_fourcc('M','J','P','G')
    #output = cv.VideoWriter('output.avi', fourcc  , int(fps) ,(int(width) , int(height)))

    # Iterating Through the frames of the video and getting the pose from them
    #success,image = vidcap.read()
    full_cc_list = []
    count = 0
    batches = 100
    startingImageIdx = 0
    while True:

        #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        images = []
        for _ in range(batches):
            if count >= amountOfFrames: continue
            #print('computing for frame: ', count)

            image = vid[count]
            images.append(image)
            count += 1
        
        pbar.update(count - startingImageIdx)
        
        resnetResults = getResultsFromResnet(images, startingImageIdx)
        startingImageIdx = count
        # Should this be but outside ??
   
        manager = Manager()
        cc_list = manager.list()

        pool_obj = multiprocessing.Pool()
        # We got to add the manager list
        pool_obj.map( getCC, [ (*result, cc_list) for result in resnetResults]  )
        #pool_obj.map(getCC, resnetResults)
    
        # The next line is necessary >_<
        time.sleep(2)
        pool_obj.close() 
        pool_obj.join()    
    
    
        # using cc list to fill out the array
        for el in cc_list:
            frameIdx, circleIdx, pt1, cc = el
            fishData[frameIdx, circleIdx, ...] = pt1
            ccData[frameIdx, circleIdx] = cc

        #keypointsList, ccList = get_pose_from_frame(image)
        #full_cc_list = full_cc_list + ccList
        #rgbIm = np.stack((image, image, image), axis = 2)
        #rbgIm = drawKeypointsList(keypointsList, rgbIm)
        #output.write( image )

        #cv.imwrite('frame0.png', im_with_pose)

        #success,image = vidcap.read()
    
        #count += 1
        #if count >= 10:
        #    break

        if count >= amountOfFrames - 1:
            break
    

    return fishData, ccData










