import numpy as np
import xlsxwriter
import math

def array2XL(array, ccData, outputPath, threshold = .4):
    workbook = xlsxwriter.Workbook(outputPath)
    worksheet = workbook.add_worksheet()

    worksheet.write(1,0,'Frame #')
    amountOfFrames = len(array)
    for frameIdx in range(amountOfFrames):
        worksheet.write(frameIdx + 2,0, frameIdx + 1)
    amoutOfFish = array.shape[1]

    step = 3
    # Writing the title of the fish
    for fishIdx in range(amoutOfFish):
        realIdx = fishIdx * step
        worksheet.write(0, realIdx + 1, 'Fish ' + str(fishIdx + 1))
        worksheet.write(1, realIdx + 1, 'Vx')
        worksheet.write(1, realIdx + 2, 'Vy')
        worksheet.write(1, realIdx + 3, 'Speed')

    for frameIdx in range(amountOfFrames):
        for fishIdx in range(amoutOfFish):
            realIdx = fishIdx * step
            vxn, _, vyn, __, length = array[frameIdx, fishIdx, ...]
            cc = ccData[frameIdx, fishIdx]
            if math.isnan(cc) or cc < threshold or np.any(np.isnan([vxn, vyn, length])): continue
            worksheet.write(frameIdx + 2, realIdx + 1, vxn * length)
            worksheet.write(frameIdx + 2, realIdx + 2, vyn * length)
            worksheet.write(frameIdx + 2, realIdx + 3, length)

    workbook.close()






