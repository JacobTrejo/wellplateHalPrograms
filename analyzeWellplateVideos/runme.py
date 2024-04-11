from programs.computeForVideo import getCorrelationCoefficientsFromVideo 
import numpy as np
from programs.bgsub import bgsubFolder
import os
from programs.fishData2Velocities import fishData2Velocity
import argparse
from programs.velocity2XL import array2XL
from programs.Config import Config
import xlsxwriter
import math

# Some analysis functions
from programs.analysisFunctions import interpolatePts, regressionSmoothing

def array2RegVelocity(fishData, ccData, fps = 50, windowSize = 5):
    offsetFromCenter = int((windowSize - 1) // 2)
    COMVs = []

    for fishIdx in range(fishData.shape[1]):

        Xs = fishData[:, fishIdx, 0, 2]
        Ys = fishData[:, fishIdx, 1, 2]

        vx, vy = regressionSmoothing(fishData[:,fishIdx,:,2], fishIdx, ccData , windowSize, fps )

        length = (vx ** 2 + vy ** 2) ** .5
        vx[offsetFromCenter:-offsetFromCenter] =\
            vx[offsetFromCenter:-offsetFromCenter] / length[offsetFromCenter:-offsetFromCenter]

        vy[offsetFromCenter:-offsetFromCenter] = \
            vy[offsetFromCenter:-offsetFromCenter] / length[offsetFromCenter:-offsetFromCenter]

        COMVs.append((vx, vy, length))

    return COMVs

#def velocityArray2XL(array, ccData, outputPath, threshold = .4):
#    workbook = xlsxwriter.Workbook(outputPath)
#    worksheet = workbook.add_worksheet()
#
#    worksheet.write(1,0,'Frame #')
#    amountOfFrames = len(array)
#    amountOfFrames = array.shape[2]
#    for frameIdx in range(amountOfFrames):
#        worksheet.write(frameIdx + 2,0, frameIdx + 1)
#    amoutOfFish = array.shape[1]
#    amoutOfFish = array.shape[0]
#
#    step = 3
#    # Writing the title of the fish
#    for fishIdx in range(amoutOfFish):
#        realIdx = fishIdx * step
#        worksheet.write(0, realIdx + 1, 'Fish ' + str(fishIdx + 1))
#        worksheet.write(1, realIdx + 1, 'Vx')
#        worksheet.write(1, realIdx + 2, 'Vy')
#        worksheet.write(1, realIdx + 3, 'Speed')
#
#    for frameIdx in range(amountOfFrames):
#        for fishIdx in range(amoutOfFish):
#            realIdx = fishIdx * step
#            vxn, vyn, length = array[fishIdx, :, frameIdx]
#            cc = ccData[frameIdx, fishIdx]
#            if math.isnan(cc) or cc < threshold or np.any(np.isnan([vxn, vyn, length])): continue
#            worksheet.write(frameIdx + 2, realIdx + 1, vxn * length)
#            worksheet.write(frameIdx + 2, realIdx + 2, vyn * length)
#            worksheet.write(frameIdx + 2, realIdx + 3, length)
#
#    workbook.close()

def velocityArray2XL(array, ccData, outputPath, threshold = .7):
    workbook = xlsxwriter.Workbook(outputPath)
    amountOfFrames = len(array)
    amountOfFrames = array.shape[2]
    worksheetSize = 500
    amountOfWorkSheets = int(np.ceil(amountOfFrames / worksheetSize))

    for worksheetIdx in range(amountOfWorkSheets):
        startingFrameIdx = worksheetSize * worksheetIdx
        endFrameIdx = worksheetSize * (worksheetIdx + 1)
        endFrameIdx = np.clip(endFrameIdx,0, amountOfFrames)
        amountOfFramesInstance = endFrameIdx - startingFrameIdx

        worksheet = workbook.add_worksheet()

        worksheet.write(1,0,'Frame #')

        for frameIdx in range(amountOfFramesInstance):
            worksheet.write(frameIdx + 2,0, frameIdx + 1 + startingFrameIdx)
        amoutOfFish = array.shape[1]
        amoutOfFish = array.shape[0]

        step = 3
        # Writing the title of the fish
        for fishIdx in range(amoutOfFish):
            realIdx = fishIdx * step
            worksheet.write(0, realIdx + 1, 'Fish ' + str(fishIdx + 1))
            worksheet.write(1, realIdx + 1, 'Vx')
            worksheet.write(1, realIdx + 2, 'Vy')
            worksheet.write(1, realIdx + 3, 'Speed')

        for frameIdx in range(amountOfFramesInstance):
            for fishIdx in range(amoutOfFish):
                realIdx = fishIdx * step
                vxn, vyn, length = array[fishIdx, :, frameIdx + startingFrameIdx]
                [vxn, vyn, length] = np.nan_to_num([vxn, vyn, length])

                cc = ccData[frameIdx + startingFrameIdx, fishIdx]
                # if math.isnan(cc) or cc < threshold or np.any(np.isnan([vxn, vyn, length])): continue
                if math.isnan(cc) or cc < threshold: continue
                if np.any(np.isnan([vxn, vyn, length])):
                    [vxn, vyn, length] = np.nan_to_num([vxn, vyn, length])

                worksheet.write(frameIdx + 2, realIdx + 1, vxn * length)
                worksheet.write(frameIdx + 2, realIdx + 2, vyn * length)
                worksheet.write(frameIdx + 2, realIdx + 3, length)

    workbook.close()




# Checking if you passed any arguments to the script
parser = argparse.ArgumentParser()
parser.add_argument('-t','--threshold',default=.7, type=float, help='the threshold which will be used to discard bad estimates')
parser.add_argument('-f','--fps',default=50, type=int, help='frames per second in which the videos were recorded in')
parser.add_argument('-w','--window',default=5, type=int, help='smoothing window, must be odd')
args = vars(parser.parse_args())
threshold = args['threshold']
fps = args['fps']
window = args['window']

print(f"Your threshold is {threshold}")

if threshold < -1 or threshold >= 1:
    print("You entered an invalid threshold")
    exit()

if fps <= 0:
    print("You entered an invalid fps")
    exit()

outputsFolder = 'outputs/'

# Lets find the number of videos in the folder
videoFolder = 'uploadYourVideosHere/' 
videoNames = os.listdir(videoFolder)

for videoName in videoNames:
    # We are currenly assuing that we have 'videos' made up of only images
    # Lets the the bgsug array of the video
    # NOTE: you might have to deal differently is and actual video is passed in
    print('computing for : ', videoFolder + videoName)
    bgsubVideo = bgsubFolder(videoFolder + videoName)
    bgsubVideo = np.array(bgsubVideo)
    fishData, ccData = getCorrelationCoefficientsFromVideo(bgsubVideo)
    fishVelocities = fishData2Velocity(fishData, ccData, threshold, fps)
    # Lets create the fish velocities
    
    # Lets interpolate the fish data
    interpacFishData = np.zeros(fishData.shape)
    interpacFishData[..., 10:] = fishData[..., 10:]
    amountOfFrames, amountOfFish = fishData.shape[:2] 
    for frameIdx in range( amountOfFrames ):
        for fishIdx in range( amountOfFish  ):
            if ccData[frameIdx, fishIdx] <  threshold:
                # Let erase the eyes we had put in
                interpacFishData[frameIdx, fishIdx] = 0
                continue
            interpPoints = interpolatePts(fishData[frameIdx, fishIdx, ...])
            interpacFishData[frameIdx, fishIdx, :, :10] = interpPoints[:, :10]
    
    COMVs = array2RegVelocity(interpacFishData, ccData, fps, window) 
    COMVs = np.array(COMVs)
    # print('comvs shape: ', COMVs.shape) #(42, 3, 201)
    # Lets get rid off the parts were we cannot smooth
    offsetFromCenter = (window - 1) // 2
    COMVs = COMVs[...,offsetFromCenter:-offsetFromCenter]

    # lets make a directory to save the fishData and ccData, and perhaps later add more stuff
    # NOTE: you might have to deal differently is and actual video is passed in
    videosOutputFolder = outputsFolder + videoName + '/'
    if not os.path.isdir(videosOutputFolder):
        os.mkdir(videosOutputFolder)
    np.save(videosOutputFolder + 'fishData.npy', fishData)
    np.save(videosOutputFolder + 'ccData.npy', ccData)
    np.save(videosOutputFolder + 'velocityData.npy', fishVelocities)
    
    # temporary
    np.save(videosOutputFolder + 'COMVs.npy', COMVs)

    #array2XL(fishVelocities, ccData, videosOutputFolder + 'velocityData.xlxs', threshold)
    velocityArray2XL(COMVs, ccData[offsetFromCenter:-offsetFromCenter], videosOutputFolder + 'velocityData.xlxs', threshold)
##arrayPath = 'inputs/bgsub_videos/testImagingBsub.npy'
#arrayPath = 'inputs/bgsub_videos/test_imaging_1-24-24_bgsub.npy'
#array = np.load(arrayPath)
#print('array shape: ', array.shape)

## Creating a short array
#shortArray = array[:1000,...]
#np.save('shortArray.npy', shortArray)

#fishData, ccData = getCorrelationCoefficientsFromVideo( 'shortArray.npy' )

#np.save('fishData_new.npy', fishData)
#np.save('ccData_new.npy', ccData)









