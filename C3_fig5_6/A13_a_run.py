import sys
sys.path.append("/sdf/home/w/winnicki/btx/")

from btx.processing.freqdir import *

from mpi4py.MPI import COMM_WORLD as comm

import csv

import matplotlib.pyplot as plt
import numpy as np


from datetime import datetime
if comm.rank==0:
    currRun = datetime.now().strftime("%y%m%d%H%M%S")
else:
    currRun = None
currRun = comm.bcast(currRun, root=0)
import time

import os

# for iirun in range(210, 306, 1):

skipNums = [229, 231, 232, 248, 275, 305, 306, 309]

for iirun in range(214, 215, 1):
    if iirun in skipNums:
        print(f"SKIPPING RUN {iirun} \n\n\n")
        continue
    else:
        print(f"\n\n\n****************** PROCESSING RUN {iirun} ******************")

    # #JOHN CHANGE 05/13/2024
    # exp = 'xppc00121'
    # # run = 511 #JOHN CHANGE 03/01/2024
    # run = 510
    # det_type = 'XppEndstation.0:Alvium.1'

    # # exp = 'xpplx9221'
    # # run = 244
    # # det_type = 'XppEndstation.0:Alvium.1'

    det_type = 'zyla_0'
    exp = 'xppx22715'
    run = iirun

    grabImgSteps = 16
    writeToHere = "/sdf/data/lcls/ds/xpp/xppx22715/scratch/winnicki/h5writes_20240521/"
    start_offset=0
    num_imgs = 12000
    alpha=0.2
    rankAdapt=False
    rankAdaptMinError = 100
    downsample = False
    bin_factor = 1
    eluThreshold = False
    eluAlpha = 0.01
    normalizeIntensity=True
    noZeroIntensity=True
    minIntensity=10000
    samplingFactor=1
    divBy=2
    threshold=True
    thresholdQuantile=0.95
    # thresholdQuantile=0.99
    unitVar = False
    usePSI = True

    roiLen = 1000
    downsampleImg = 400
    thumbLen = 64
    centerImg = True

    finalWrites = []
    if not usePSI:
        dataGen = WrapperFullFD(exp = exp, run = run,
                det_type = det_type, 
                grabImgSteps = grabImgSteps,
                writeToHere = writeToHere, 
                start_offset=start_offset, num_imgs=num_imgs,  
                num_components=200, alpha=alpha, rankAdapt=rankAdapt, rankAdaptMinError = rankAdaptMinError,
                downsample=downsample, bin_factor=bin_factor, eluThreshold=eluThreshold, eluAlpha=eluAlpha,
                threshold=threshold, normalizeIntensity=normalizeIntensity, 
                noZeroIntensity=noZeroIntensity, minIntensity = minIntensity,
                samplingFactor=samplingFactor, divBy=divBy, thresholdQuantile=thresholdQuantile, unitVar = unitVar,
                usePSI=usePSI, roiLen=roiLen, downsampleImg=downsampleImg, thumbLen=thumbLen, centerImg = centerImg)
        dataGen.oldCompDecayingSVD2(int(currRun[:-5]), 10000, 10500) #JOHN CHANGE 06/27/2024: Use this for scaling test. 
        # dataGen.compDecayingSVD(int(currRun[:-5]), 200, 6400) #JOHN CHANGE 06/27/2024: Use this for parallel test. 

    numComponents = 100
    keepArr = []

    comm.barrier()
    wrappingTest = WrapperFullFD(exp = exp, run = run,
            det_type = det_type, 
            grabImgSteps = grabImgSteps,
            writeToHere = writeToHere, 
            start_offset=start_offset, num_imgs=num_imgs,  
            num_components=numComponents, alpha=alpha, rankAdapt=rankAdapt, rankAdaptMinError = rankAdaptMinError,
            downsample=downsample, bin_factor=bin_factor, eluThreshold=eluThreshold, eluAlpha=eluAlpha,
            threshold=threshold, normalizeIntensity=normalizeIntensity, 
            noZeroIntensity=noZeroIntensity, minIntensity = minIntensity,
            samplingFactor=samplingFactor, divBy=divBy, thresholdQuantile=thresholdQuantile, unitVar = unitVar,
            usePSI=usePSI, roiLen=roiLen, downsampleImg=downsampleImg, thumbLen=thumbLen, centerImg = centerImg)
    st = time.process_time() 
    if usePSI:
        wrappingTest.retrieveImages()
    else:
        wrappingTest.fullImgData = dataGen.fullImgData
        wrappingTest.imgsTracked = dataGen.imgsTracked
    perftime = wrappingTest.runMe()
    et = time.process_time()
    # print("TESTING RECON ERROR:", wrappingTest.fullImgData.T.shape, wrappingTest.matSketch.shape)
    # reconError = wrappingTest.lowMemoryReconstructionErrorScaled(wrappingTest.fullImgData.T, wrappingTest.matSketch[:wrappingTest.num_components, :])
    # print("FULL RECON ERROR: ", reconError)
    #         #    wrappingTest.visualizeMe()
    # field = [et-st, perftime, reconError]
    # keepArr.append(field)

    # finProcessTime = sum([j[0] for j in keepArr])/len(keepArr)
    # finWallTime = sum([j[1] for j in keepArr])/len(keepArr)
    # finReconError = sum([j[2] for j in keepArr])/len(keepArr)

    # finProcessTime = comm.allgather(finProcessTime)
    # finWallTime = comm.allgather(finWallTime)
    # finReconError = comm.allgather(finReconError)

    # finalWrites.append([numComponents, sum(finProcessTime)/len(finProcessTime), sum(finWallTime)/len(finWallTime), sum(finReconError)/len(finReconError)])

    # print(finalWrites)

    print(f"FINISHED RUN IN {et - st} SECONDS")