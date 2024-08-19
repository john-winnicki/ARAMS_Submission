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
mypath = './scalingOutputs/'+currRun 
if comm.rank==0:
    if not os.path.isdir(mypath):
        os.makedirs(mypath)
        os.makedirs(mypath+'/figs')
    
    if os.path.isdir(f'/sdf/data/lcls/ds/mfx/mfxp23120/scratch/winnicki/h5writes_20240109/{currRun}_{comm.Get_size()}'):
        raise Exception("Directory already exists")
    else:
        os.mkdir(f'/sdf/data/lcls/ds/mfx/mfxp23120/scratch/winnicki/h5writes_20240109/{currRun}_{comm.Get_size()}')

exp = 'xppc00121'
run = 511
det_type = 'XppEndstation.0:Alvium.1'
grabImgSteps = 16
writeToHere = f'/sdf/data/lcls/ds/mfx/mfxp23120/scratch/winnicki/h5writes_20240109/{currRun}_{comm.Get_size()}'
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
minIntensity=100000
samplingFactor=1
# divBy=2 if int(comm.Get_size()) in [1, 2, 4, 8, 16, 32, 64, 128, 256] else 3
divBy=2
threshold=True
thresholdQuantile=0.9975
unitVar = False
usePSI = False

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
            usePSI=usePSI)
    dataGen.compDecayingSVD(int(currRun[:-5]), 500*16, 414720*4)

keepArr = []

for numComponents in range(1000, 1001, 1):
    with open(mypath+'/scalingTest_Time{}_Rank{}-{}_NumComponents{}.csv'.format(currRun, comm.rank, comm.size, numComponents), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Process Time", "Wall Time", "Reconstruction Error"])
        keepArr = []
        for j in range(3):
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
                    usePSI=usePSI)
            print("IMAGE RETRIEVAL STARTED")
            st = time.process_time() 
            if usePSI:
                wrappingTest.retrieveImages()
            else:
                wrappingTest.fullImgData = dataGen.fullImgData
                wrappingTest.imgsTracked = dataGen.imgsTracked
            # JOHN CHANGE 01/08/2024
            print("IMAGE RETRIEVAL ENDED")
            perftime = wrappingTest.runMe()
            print("FINISHED RUNNING")
            perfTime = wrappingTest.newBareTime
            print("PERFTIME:", perfTime)
            et = time.process_time() 
            print("TESTING RECON ERROR:", wrappingTest.fullImgData.T.shape, wrappingTest.matSketch.shape)
            # reconError = wrappingTest.lowMemoryReconstructionErrorScaled(wrappingTest.fullImgData.T, wrappingTest.matSketch[:wrappingTest.num_components, :]) #JOHN CHANGE 12/07/2024. For rank adaptive tests, this basically ruins the test since you restrict the number of components. 
            reconError = wrappingTest.lowMemoryReconstructionErrorScaled(wrappingTest.fullImgData.T, wrappingTest.matSketch)
            print("FULL RECON ERROR: ", reconError)
#            wrappingTest.visualizeMe()
            field = [et-st, perftime, reconError]
            writer.writerow(field)
            keepArr.append(field)
        finProcessTime = sum([j[0] for j in keepArr])/len(keepArr)
        finWallTime = sum([j[1] for j in keepArr])/len(keepArr)
        finReconError = sum([j[2] for j in keepArr])/len(keepArr)
        writer.writerow([])
        writer.writerow([finProcessTime, finWallTime, finReconError])
        writer.writerow([])
        finProcessTime = comm.allgather(finProcessTime)
        finWallTime = comm.allgather(finWallTime)
        finReconError = comm.allgather(finReconError)

        finalWrites.append([numComponents, sum(finProcessTime)/len(finProcessTime), sum(finWallTime)/len(finWallTime), sum(finReconError)/len(finReconError)])

        writer.writerow([])
        writer.writerow(finProcessTime)
        writer.writerow(finProcessTime)
        writer.writerow(finReconError)
        writer.writerow([])
        writer.writerow(["det_type ="+ str(det_type),
            "grabImgSteps ="+ str(grabImgSteps),
            "writeToHere ="+ str(writeToHere),
            "start_offset="+str(start_offset), "num_imgs="+str(num_imgs),
            "num_components="+str(numComponents),
            "alpha="+str(alpha), "rankAdapt="+str(rankAdapt), 
            "rankAdaptMinError ="+ str(rankAdaptMinError),
            "downsample="+str(downsample), "bin_factor="+str(bin_factor), 
            "eluThreshold="+str(eluThreshold), "eluAlpha="+str(eluAlpha),
            "threshold="+str(threshold), "normalizeIntensity="+str(normalizeIntensity),
            "noZeroIntensity="+str(noZeroIntensity), "minIntensity ="+ str(minIntensity),
            "samplingFactor="+str(samplingFactor),
            "divBy="+str(divBy), "thresholdQuantile="+str(thresholdQuantile),
            "unitVar="+str(unitVar),
            "usePSI="+str(usePSI)])

if comm.rank==0:
    x = np.array([j[0] for j in finalWrites])

    y = np.array([j[1] for j in finalWrites])
    plt.scatter(x, y)
    plt.savefig(mypath+'/figs/{}_processtime.png'.format(currRun))
    plt.clf()
    
    y = np.array([j[2] for j in finalWrites])
    plt.scatter(x, y)
    plt.savefig(mypath+'/figs/{}_walltime.png'.format(currRun))
    plt.clf()

    y = np.array([j[3] for j in finalWrites])
    plt.scatter(x, y)
    plt.savefig(mypath+'/figs/{}_reconerror.png'.format(currRun))
    plt.clf()
    with open(mypath+'/scalingTest_Time{}_{}_FINAL_VALUES.csv'.format(currRun, comm.size), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Process Time", "Wall Time", "Reconstruction Error"])
        for row in finalWrites:
            writer.writerow(row)
        writer.writerow(["det_type ="+ str(det_type),
            "grabImgSteps ="+ str(grabImgSteps),
            "writeToHere ="+ str(writeToHere),
            "start_offset="+str(start_offset), "num_imgs="+str(num_imgs),
            "num_components="+str(numComponents),
            "alpha="+str(alpha), "rankAdapt="+str(rankAdapt), 
            "rankAdaptMinError ="+ str(rankAdaptMinError),
            "downsample="+str(downsample), "bin_factor="+str(bin_factor), 
            "eluThreshold="+str(eluThreshold), "eluAlpha="+str(eluAlpha),
            "threshold="+str(threshold), "normalizeIntensity="+str(normalizeIntensity),
            "noZeroIntensity="+str(noZeroIntensity), "minIntensity ="+ str(minIntensity),
            "samplingFactor="+str(samplingFactor),
            "divBy="+str(divBy), "thresholdQuantile="+str(thresholdQuantile),
            "usePSI="+str(usePSI)])

print(f"RESULTS FOR {comm.Get_size()} CORES: {finalWrites}")

print("FINISHED TRUE RUN")