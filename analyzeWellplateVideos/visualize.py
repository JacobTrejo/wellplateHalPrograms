import numpy as np
import cv2 as cv
import os
import math
from tqdm import tqdm

outputsFolder = 'outputs/'
videoFolder = 'uploadYourVideosHere/'

fps = 50

threshold = -1

# NOTE: the algorithm currently assumes we are working with folders filled with images
videoNames = os.listdir(videoFolder)
for videoName in videoNames:
    videoPath = videoFolder + videoName
    print('rendering pose for folder: ', videoName)
    frameNames = os.listdir(videoPath)
    frameNames.sort()
    

    # The pose data
    fishData = np.load(outputsFolder + videoName + '/fishData.npy')
    ccData = np.load(outputsFolder + videoName + '/ccData.npy')
    amountOfCircles = fishData.shape[1]

    frame0 = cv.imread(videoPath + '/' + frameNames[0])
    height, width = frame0.shape[:2]

    fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    output = cv.VideoWriter(outputsFolder + videoName + '/' + videoName  + '.avi', fourcc, int(fps), (int(width), int(height)))
    
    for frameIdx, frameName in tqdm(enumerate(frameNames), total = len(frameNames)):
        frame = cv.imread(videoPath + '/'  + frameName)
        #if len(frame.shape) > 2: frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        poses = fishData[frameIdx]
        for poseIdx, pose in enumerate(poses):
    
            pose = pose.astype(int)

            if np.any( np.logical_or( np.logical_or(pose[0,:] >= width ,  pose[0,:] < 0 ), np.logical_or(  pose[1,:] >= height , pose[1,:] < 0) )): continue
            if math.isnan(ccData[frameIdx, poseIdx]) or ccData[frameIdx, poseIdx] < threshold: continue

            for pointIdx in range(10):
                frame = cv.circle(frame, (pose[0,pointIdx], pose[1,pointIdx]), 2, (0,255,0), -1)
            for pointIdx in range(10,12):
                frame = cv.circle(frame, (pose[0,pointIdx], pose[1,pointIdx]), 2, (0,0,255), -1)

        output.write(frame)
    output.release()


videoNames = os.listdir(videoFolder)
for videoName in videoNames:
    videoPath = videoFolder + videoName
    print('rendering velocities for folder: ', videoName)
    frameNames = os.listdir(videoPath)
    frameNames.sort()


    # The pose data
    fishData = np.load(outputsFolder + videoName + '/fishData.npy')
    ccData = np.load(outputsFolder + videoName + '/ccData.npy')
    velocityData = np.load(outputsFolder + videoName + '/velocityData.npy')
    
    amountOfCircles = fishData.shape[1]

    frame0 = cv.imread(videoPath + '/' + frameNames[0])
    height, width = frame0.shape[:2]

    fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    output = cv.VideoWriter(outputsFolder + videoName + '/' + videoName  + '_velocity.avi', fourcc, int(fps), (int(width), int(height)))

    
    lenghts = velocityData[...,-1]
    lenghtsMask = np.invert(np.isnan(lenghts))
    #ratio = 60 / np.max(velocityData[...,-1])    
    ratio = 60 / np.max(lenghts[lenghtsMask])
    
    # velocity is not defined for the final frame
    frameNames.pop(-1) 

    for frameIdx, frameName in tqdm(enumerate(frameNames), total = len(frameNames)):
        #if frameIdx >= 200: continue  
        frame = cv.imread(videoPath + '/'  + frameName)

        for fishIdx in range(amountOfCircles):
            
            #if np.any(np.isnan(velocityData[frameIdx, fishIdx,:])): continue
            if math.isnan(ccData[frameIdx, fishIdx]) or ccData[frameIdx, fishIdx] < threshold: continue 
            if np.any(np.isnan(velocityData[frameIdx, fishIdx,:])): 
                vxn, comx, vyn, comy, lenght = np.nan_to_num(velocityData[frameIdx,fishIdx,:] )
            else:
                vxn, comx, vyn, comy, lenght = velocityData[frameIdx,fishIdx,:]
            
            xoffset = ratio * lenght * vxn
            yoffset = ratio * lenght * vyn
            
            frame = cv.arrowedLine(frame, (int(comx), int(comy)), (int(comx + xoffset), int(comy + yoffset)), (255,255,0), 3, 5, 0, .5)
        
        output.write(frame)

    output.release()
        







