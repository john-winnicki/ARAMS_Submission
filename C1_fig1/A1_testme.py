import os, csv, argparse
import math
import time
import random
from collections import Counter

import numpy as np
from numpy import zeros, sqrt, dot, diag
import heapq

from datetime import datetime

import matplotlib.pyplot as plt

import sys

class CustomPriorityQueue:
    """
    Custom Priority Queue. 

    Maintains a priority queue of items based on user-inputted priority for said items. 
    """
    def __init__(self, max_size):
        self.queue = []
        self.index = 0  # To handle items with the same priority
        self.max_size = max_size

    def push(self, item, priority, origWeight):
        if len(self.queue) >= self.max_size:
            self.pop()  # Remove the lowest-priority item if queue is full
        heapq.heappush(self.queue, (priority, self.index, (item, priority, origWeight)))
        self.index += 1

    def pop(self):
        return heapq.heappop(self.queue)[-1]

    def is_empty(self):
        return len(self.queue) == 0

    def size(self):
        return len(self.queue)

    def get(self):
        ret = []
        while self.queue:
            curr = heapq.heappop(self.queue)[-1]
            ret.append(curr[0])
        return ret

class PrioritySampling:
    """
    Priority Sampling. 

    Based on [1] and [2]. Frequent Directions is a sampling algorithm that, 
    given a high-volume stream of weighted items, creates a generic sample 
    of a certain limited size that can later be used to estimate the total 
    weight of arbitrary subsets. In our case, we use Priority Sampling to
    generate a matrix sketch based, sampling rows of our data using the
    2-norm as weights. Priority Sampling "first assigns each element i a random 
    number u_i ∈ Unif(0, 1). This implies a priority p_i = w_i/u_i , based 
    on its weight w_i (which for matrix rows w_i = ||a||_i^2). We then simply 
    retain the l rows with largest priorities, using a priority queue of size l."

    [1] Nick Duffield, Carsten Lund, and Mikkel Thorup. 2007. Priority sampling for 
    estimation of arbitrary subset sums. J. ACM 54, 6 (December 2007), 32–es. 
    https://doi.org/10.1145/1314690.1314696

    Attributes
    ----------
    ell: Number of components to keep
    d: Number of features of each datapoint
    sketch: Matrix Sketch maintained by Priority Queue

    """
    def __init__(self, ell, d):
        self.ell = ell
        self.d = d
        self.sketch = CustomPriorityQueue(self.ell)

    def update(self, vec):
        ui = random.random()
        wi = np.linalg.norm(vec)**2
        pi = wi/ui
        self.sketch.push(vec, pi, wi)






class FreqDir:

    """
    Parallel Rank Adaptive Frequent Directions.
    
    Based on [1] and [2]. Frequent Directions is a matrix sketching algorithm used to
    approximate large data sets. The basic goal of matrix sketching is to process an
    n x d matrix A to somehow represent a matrix B so that ||A-B|| or covariance error
    is small. Frequent Directions provably acheives a spectral bound on covariance 
    error and greatly outperforms comparable existing sketching techniques. It acheives
    similar runtime and performance to incremental SVD as well. 

    In this module we implement the frequent directions algorithm. This is the first of
    three modules in this data processing pipeline, and it produces a sketch of a subset
    of the data into an h5 file. The "Merge Tree" module will be responsible for merging
    each of the sketches together, parallelizing the process, and the apply compression
    algorithm will be responsible for using the full matrix sketch projecting the 
    original data to low dimensional space for data exploration. 

    One novel feature of this implementation is the rank adaption feature: users have the
    ability to select the approximate reconstruction error they want the sketch to operate
    over, and the algorithm will adjust the rank of the sketch to meet this error bound
    as data streams in. The module also gives users the ability to perform the sketching
    process over thresholded and non-zero image data.

    [1] Frequent Directions: Simple and Deterministic Matrix 
    Sketching Mina Ghashami, Edo Liberty, Jeff M. Phillips, and 
    David P. Woodruff SIAM Journal on Computing 2016 45:5, 1762-1792

    [2] Ghashami, M., Desai, A., Phillips, J.M. (2014). Improved 
    Practical Matrix Sketching with Guarantees. In: Schulz, A.S., 
    Wagner, D. (eds) Algorithms - ESA 2014. ESA 2014. Lecture Notes 
    in Computer Science, vol 8737. Springer, Berlin, Heidelberg. 
    https://doi.org/10.1007/978-3-662-44777-2_39

    Attributes
    ----------
       start_offset: starting index of images to process
       num_imgs: total number of images to process
       ell: number of components of matrix sketch
       alpha: proportion  of components to not rotate in frequent directions algorithm
       exp, run, det_type: experiment properties
       rankAdapt: indicates whether to perform rank adaptive FD
       increaseEll: internal variable indicating whether ell should be increased for rank adaption
       output_dir: directory to write output
       merger: indicates whether object will be used to merge other FD objects
       mergerFeatures: used if merger is true and indicates number of features of local matrix sketches
       downsample, bin_factor: whether data should be downsampled and by how much
       threshold: whether data should be thresholded (zero if less than threshold amount)
       normalizeIntensity: whether data should be normalized to have total intensity of one
       noZeroIntensity: whether data with low total intensity should be discarded
       d: number of features (pixels) in data
       m: internal frequent directions variable recording total number of components used in algorithm
       sketch: numpy array housing current matrix sketch
       mean: geometric mean of data processed
       num_incorporated_images: number of images processed so far
       imgsTracked: indices of images processed so far
       currRun: Current datetime used to identify run
       samplingFactor: Proportion of batch data to process based on Priority Sampling Algorithm
    """

    def __init__(
        self,
        comm,
        rank,
        size,
        start_offset,
        num_imgs,
        exp,
        run,
        det_type,
        output_dir,
        currRun,
        imgData,
        imgsTracked,
        alpha,
        rankAdapt,
        rankAdaptMinError,
        merger,
        mergerFeatures,
        downsample,
        bin_factor,
        samplingFactor, 
        num_components,
        psi,
        usePSI
    ):


########################
        self.start_offset = start_offset
        self.downsample = False
        self.bin_factor = 0
        self.output_dir = output_dir
        self.num_components = num_components
        self.num_features,self.num_images = imgData.shape 

        self.task_durations = dict({})

        self.num_incorporated_images = 0
        self.outliers, self.pc_data = [], []
########################
        self.comm = comm
        self.rank= rank
        self.size = size

        self.currRun = currRun

        self.output_dir = output_dir

        self.merger = merger

        if self.merger:
            self.num_features = mergerFeatures

        self.num_incorporated_images = 0

        self.d = self.num_features
        self.ell = num_components
        self.m = 2*self.ell
        self.sketch = zeros( (self.m, self.d) ) 
        self.nextZeroRow = 0
        self.alpha = alpha

        self.rankAdapt = rankAdapt
        self.rankAdaptMinError = rankAdaptMinError
        self.increaseEll = False

        self.samplingFactor = samplingFactor

        self.imgData = imgData
        self.imgsTracked = imgsTracked

        self.fdTime = 0

    def run(self):
        """
        Perform frequent directions matrix sketching
        on run subject to initialization parameters.
        """
        img_batch = self.imgData
        if self.samplingFactor<1:
            st = time.process_time() 
            psamp = PrioritySampling(int((img_batch.shape[1])*self.samplingFactor), self.d)
            for row in img_batch.T:
                psamp.update(row)
            img_batch = np.array(psamp.sketch.get()).T
            et = time.process_time() 
            self.fdTime += et - st
        self.update_model(img_batch)

    def update_model(self, X):
        """
        Update matrix sketch with new batch of observations. 

        The matrix sketch array is of size 2*ell. The first ell rows maintained
        represent the current matrix sketch. The next ell rows form a buffer.
        Each row of the data is added to the buffer until ell rows have been
        accumulated. Then, we apply the rotate function to the buffer, which
        incorporates the buffer data into the matrix sketch. 
        
        Following the rotation step, it is checked if rank adaption is enabled. Then,
        is checked if there is enough data to perform one full rotation/shrinkage
        step. Without this check, one runs the risk of having zero rows in the
        sketch, which is innaccurate in representing the data one has seen.
        If one can increase the rank, the increaseEll flag is raised, and once sufficient
        data has been accumulated in the buffer, the sketch and buffer size is increased.
        This happens when we check if increaseEll, canRankAdapt, and rankAdapt are all true,
        whereby we check if we should be increasing the rank due to high error, we
        have sufficient incoming data to do so (to avoid zero rows in the matrix sketch), 
        and the user would like for the rank to be adaptive, respectively. 
        
        Parameters
        ----------
        X: ndarray
            data to update matrix sketch with
        """

        rankAdapt_increaseAmount = 20

        _, numIncorp  = X.shape
        origNumIncorp = numIncorp
        # with TaskTimer(self.task_durations, "total update"):
        if self.rank==0 and not self.merger:
            print(
                "Factoring {m} sample{s} into {n} sample, {q} component model...".format(
                    m=numIncorp, s="s" if numIncorp > 1 else "", n=self.num_incorporated_images, q=self.ell
                )
            )
        for row in X.T:
            st = time.process_time() 
            canRankAdapt = numIncorp > (self.ell + rankAdapt_increaseAmount)
            if self.nextZeroRow >= self.m:
                if self.increaseEll and canRankAdapt and self.rankAdapt:
                    self.ell = self.ell + rankAdapt_increaseAmount
                    self.m = 2*self.ell
                    self.sketch = np.vstack((*self.sketch, np.zeros((2*rankAdapt_increaseAmount, self.d))))
                    self.increaseEll = False
                    print("Increasing rank of process {} to {}".format(self.rank, self.ell))
                else:
                    copyBatch = self.sketch[self.ell:,:].copy()
                    self.rotate()
                    if canRankAdapt and self.rankAdapt:
                        reconError = np.sqrt(self.lowMemoryReconstructionErrorScaled(copyBatch))
                        print("RANK ADAPT RECON ERROR: ", reconError)
                        if (reconError > self.rankAdaptMinError):
                            self.increaseEll = True
            self.sketch[self.nextZeroRow,:] = row 
            self.nextZeroRow += 1
            self.num_incorporated_images += 1
            numIncorp -= 1
            et = time.process_time() 
            self.fdTime += et - st
    
    def rotate(self):
        """ 
        Apply Frequent Directions rotation/shrinkage step to current matrix sketch and adjoined buffer. 

        The Frequent Directions algorithm is inspired by the well known Misra Gries Frequent Items
        algorithm. The Frequent Items problem is informally as follows: given a sequence of items, find the items which occur most frequently. The Misra Gries Frequent Items algorithm maintains a dictionary of <= k items and counts. For each item in a sequence, if the item is in the dictionary, increase its count. if the item is not in the dictionary and the size of the dictionary is <= k, then add the item with a count of 1 to the dictionary. Otherwise, decrease all counts in the dictionary by 1 and remove any items with 0 count. Every item which occurs more than n/k times is guaranteed to appear in the output array.

        The Frequent Directions Algorithm works in an analogous way for vectors: in the same way that Frequent Items periodically deletes ell different elements, Frequent Directions periodically "shrinks? ell orthogonal vectors by roughly the same amount. To do so, at each step: 1) Data is appended to the matrix sketch (whereby the last ell rows form a buffer and are zeroed at the start of the algorithm and after each rotation). 2) Matrix Sketch is rotated from left via SVD so that its rows are orthogonal and in descending magnitude order. 3) Norm of sketch rows are shrunk so that the smallest direction is set to 0.

        This function performs the rotation and shrinkage step by performing SVD and left multiplying by the unitary U matrix, followed by a subtraction. This particular implementation follows the alpha FD algorithm, which only performs the shrinkage step on the first alpha rows of the sketch, which has been shown to perform better than vanilla FD in [2]. 

        Notes
        -----
        Based on [1] and [2]. 

        [1] Frequent Directions: Simple and Deterministic Matrix 
        Sketching Mina Ghashami, Edo Liberty, Jeff M. Phillips, and 
        David P. Woodruff SIAM Journal on Computing 2016 45:5, 1762-1792

        [2] Ghashami, M., Desai, A., Phillips, J.M. (2014). Improved 
        Practical Matrix Sketching with Guarantees. In: Schulz, A.S., 
        Wagner, D. (eds) Algorithms - ESA 2014. ESA 2014. Lecture Notes 
        in Computer Science, vol 8737. Springer, Berlin, Heidelberg. 
        https://doi.org/10.1007/978-3-662-44777-2_39
        """
        [_,S,Vt] = np.linalg.svd(self.sketch , full_matrices=False)
        ssize = S.shape[0]
        if ssize >= self.ell:
            sCopy = S.copy()
           #JOHN: I think actually this should be ell+1 and ell. We lose a component otherwise.
            toShrink = S[:self.ell]**2 - S[self.ell-1]**2
            #John: Explicitly set this value to be 0, since sometimes it is negative
            # or even turns to NaN due to roundoff error
            toShrink[-1] = 0
            toShrink = sqrt(toShrink)
            toShrink[:int(self.ell*(1-self.alpha))] = sCopy[:int(self.ell*(1-self.alpha))]
            #self.sketch[:self.ell:,:] = dot(diag(toShrink), Vt[:self.ell,:]) #JOHN: Removed this extra colon 10/01/2023
            self.sketch[:self.ell,:] = dot(diag(toShrink), Vt[:self.ell,:])
            self.sketch[self.ell:,:] = 0
            self.nextZeroRow = self.ell
        else:
            self.sketch[:ssize,:] = diag(s) @ Vt[:ssize,:]
            self.sketch[ssize:,:] = 0
            self.nextZeroRow = ssize

    def reconstructionError(self, matrixCentered):
        """ 
        Compute the reconstruction error of the matrix sketch
        against given data

        Parameters
        ----------
        matrixCentered: ndarray
           Data to compare matrix sketch to 

        Returns
        -------
        float,
            Data subtracted by data projected onto sketched space, scaled by minimum theoretical sketch
       """
        matSketch = self.sketch
        k = 10
        matrixCenteredT = matrixCentered.T
        matSketchT = matSketch.T
        U, S, Vt = np.linalg.svd(matSketchT)
        G = U[:,:k]
        UA, SA, VtA = np.linalg.svd(matrixCenteredT)
        UAk = UA[:,:k]
        SAk = np.diag(SA[:k])
        VtAk = VtA[:k]
        Ak = UAk @ SAk @ VtAk
        return (np.linalg.norm(
        	matrixCenteredT - G @ G.T @ matrixCenteredT, 'fro')**2)/(
                (np.linalg.norm(matrixCenteredT - Ak, 'fro'))**2) 


    def lowMemoryReconstructionErrorScaled(self, matrixCentered):
        matSketch = self.sketch[:self.ell, :]
        matrixCenteredT = matrixCentered.T
        matSketchT = matSketch.T
        U, S, Vt = np.linalg.svd(matSketchT, full_matrices=False)
        G = U
        return self.estimFrobNormJ(matrixCenteredT, [G,G.T,matrixCenteredT], 20)/np.linalg.norm(matrixCenteredT, 'fro')

    def estimFrobNormJ(self, addMe, arrs, k):
        m, n = addMe.shape
        randMat = np.random.normal(0, 1, size=(n, k))
        minusMe = addMe @ randMat
        sumMe = 0
        for arr in arrs[::-1]:
            randMat = arr @ randMat
        sumMe += math.sqrt(1/k) * np.linalg.norm(randMat - minusMe, 'fro')
        return sumMe




# New Data
def genNewData(seedMe, a, b):
    numFeats = a
    numSamps = b
    perturbation = np.random.rand(numSamps, numFeats)*0.1
    np.random.seed(seedMe)
    A1 = np.random.rand(numSamps, numFeats) 
    Q1, R1 = np.linalg.qr(A1)
    Q1 = Q1 + perturbation
    A2 = np.random.rand(numFeats, numFeats) #Modify
    Q2, R2 = np.linalg.qr(A2)
    S = list(np.random.rand(numFeats)) #Modify
    S.sort()
    S = S[::-1]
    for j in range(len(S)): #Modify
        S[j] = (2**(-16*(j+1)/len(S)))*S[j] #SCALING RUN
    return((Q1 @ np.diag(S) @ Q2).T, [(0, numSamps)])

def lowMemoryReconstructionErrorScaled(matrixCentered, sketch, ell):
        matSketch = sketch[:ell, :]
        matrixCenteredT = matrixCentered.T
        matSketchT = matSketch.T
        U, S, Vt = np.linalg.svd(matSketchT, full_matrices=False)
        G = U
        return estimFrobNormJ(matrixCenteredT, [G,G.T,matrixCenteredT], 20)/np.linalg.norm(matrixCenteredT, 'fro')

def estimFrobNormJ(addMe, arrs, k):
    m, n = addMe.shape
    randMat = np.random.normal(0, 1, size=(n, k))
    minusMe = addMe @ randMat
    sumMe = 0
    for arr in arrs[::-1]:
        randMat = arr @ randMat
    sumMe += math.sqrt(1/k) * np.linalg.norm(randMat - minusMe, 'fro')
    return sumMe








#### Generate error tolerances 
import numpy as np

def generate_cubic_points(start, end, num_points):
    x_max = start ** (1/3)
    x_min = end ** (1/3)
    x_values = np.linspace(x_max, x_min, num_points)
    y_values = x_values ** 3
    return y_values.tolist()

def genEvenlySpacedLogPoints(a, b, num_points):
        """
        Generate points evenly dispersed in log space within a specified linear range.

        Parameters:
        a (float): The start of the range in linear space.
        b (float): The end of the range in linear space.
        num_points (int): The number of points to generate.

        Returns:
        np.ndarray: An array of points evenly dispersed in log space within the specified range.
        """
        log_a = np.log10(a)
        log_b = np.log10(b)
        log_space_points = np.linspace(log_a, log_b, num_points)
        linear_space_points_within_range = 10**log_space_points
        return linear_space_points_within_range
















################################################################
####### MAIN RUNNING SEQUENCE ##################################
################################################################

def main():
    if len(sys.argv) < 4:
        print("Usage: python script.py <string> <boolean> <float>")
        return
    try:
        decayType = str(sys.argv[1])
        rankAdapt = sys.argv[2].lower() in ['true', '1', 't', 'y', 'yes']
        samplingFactor = float(sys.argv[3])
    except ValueError:
        print("Error: Please provide a valid string, boolean, and float.")
        return
    print(f"decayType: {decayType}")
    print(f"rankAdapt: {rankAdapt}")
    print(f"samplingFactor: {samplingFactor}")
    
    currTime = datetime.now().strftime("%y%m%d%H%M%S")
    currRun = f"{currTime}_{rankAdapt}_{str(samplingFactor).replace('.', '-')}_{decayType}"


    mypath = './scalingOutputs/'+f"{decayType}_" + currTime 
    if not os.path.isdir(mypath):
        os.makedirs(mypath)
    exp = 'xppc00121'
    run = 511
    det_type = 'XppEndstation.0:Alvium.1'
    grabImgSteps = 16
    writeToHere = f'./'
    start_offset=0
    num_imgs = 12000
    # alpha=0.2 #JOHN RAN NEW EXPERIMENT 08/14/2024 WITH 0.4 INSTEAD OF 0.2
    alpha = 0.4
    # rankAdapt=True
    rankAdaptMinError = 0.15 #THIS IS NO LONGER USED IN THIS CODE
    downsample = False
    bin_factor = 1
    eluThreshold = False
    eluAlpha=0.01
    normalizeIntensity=True
    noZeroIntensity=True
    minIntensity=10000
    # samplingFactor=0.2
    divBy=2
    threshold=True
    thresholdQuantile=0.99
    unitVar = False
    usePSI = False

    # finalWrites = []
    imgsTracked = [(0, 1)]
    fullImgData = np.load(f'/sdf/home/w/winnicki/papertests_20240717/expDecayingSingularValues_{decayType}.npy')
    # fullImgData = np.load('/sdf/home/w/winnicki/papertests_20240717/cubicallyDecayingSingularValues_small.npy')
    # fullImgData = np.load('/sdf/home/w/winnicki/papertests_20240717/cubicallyDecayingSingularValues.npy')
    # for numComponents in range(10, 500, 30):
    #07/25/2024
    plotMe = []
    tempFileName = f"{mypath}/tempErrorVsTime_{decayType}_{rankAdapt}_{str(samplingFactor).replace('.', '-')}.npy"

    # if decayType=="bot":
    #     if samplingFactor==1.0: 
    #         listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #         if rankAdapt:
    #             listOfRanks = [5 for _ in listOfRanks]
    #             # errTols = generate_cubic_points(1.0, 0.05, len(listOfRanks))
    #             errTols = np.linspace(1.0, 0.1, len(listOfRanks))
    #         else:
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    #     else:
    #         listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #         if rankAdapt:
    #             listOfRanks = [5 for _ in listOfRanks]
    #             # errTols = generate_cubic_points(1.0, 0.05, len(listOfRanks))
    #             errTols = np.linspace(1.0, 0.01, len(listOfRanks))
    #         else:
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    # else:
    #     listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #     if rankAdapt:
    #         listOfRanks = [5 for _ in listOfRanks]
    #         # errTols = generate_cubic_points(1.0, 0.05, len(listOfRanks))
    #         errTols = np.linspace(1.0, 0.01, len(listOfRanks))
    #     else:
    #         listOfRanks = [x for x in listOfRanks]
    #         errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    
    #Agenda
    # Top: Keep both non-RA the same. Keep RA non-PS the same. Increase error tolerance for RA PS. 
    # Middle: Keep both non-RA the same. Keep RA non-PS the same. Increase error tolerance for RA PS. 
    # Bottom: Keep both non-RA the same. Increase error tolerance for both non-RA. 

    #tempAnalysis7
    # if decayType == "top":
    #     if samplingFactor == 1.0: #non-PS
    #         if rankAdapt: #keep RA non-PS the same
    #             listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #             listOfRanks = [5 for _ in listOfRanks] 
    #             errTols = np.linspace(1.0, 0.1, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    #     else: 
    #         listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #         if rankAdapt:
    #             listOfRanks = [5 for _ in listOfRanks]
    #             errTols = np.linspace(1.0, 0.0001, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    # if decayType == "mid":
    #     if samplingFactor == 1.0: #non-PS
    #         if rankAdapt: #keep RA non-PS the same
    #             listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #             listOfRanks = [5 for _ in listOfRanks] 
    #             errTols = np.linspace(1.0, 0.1, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    #     else: 
    #         listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #         if rankAdapt:
    #             listOfRanks = [5 for _ in listOfRanks]
    #             errTols = np.linspace(1.0, 0.0001, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    # if decayType == "bot":
    #     if samplingFactor == 1.0: #non-PS
    #         if rankAdapt: #keep RA non-PS the same
    #             listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #             listOfRanks = [5 for _ in listOfRanks] 
    #             errTols = np.linspace(1.0, 0.0001, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    #     else: 
    #         listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #         if rankAdapt:
    #             listOfRanks = [5 for _ in listOfRanks]
    #             errTols = np.linspace(1.0, 0.0001, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption

# #TempAnalysis9
#     if decayType == "top":
#         if samplingFactor == 1.0: #non-PS
#             if rankAdapt: #keep RA non-PS the same
#                 listOfRanks = np.linspace(5, 500, 50, dtype=int)
#                 listOfRanks = [5 for _ in listOfRanks] 
#                 errTols = genEvenlySpacedLogPoints(1.0, 0.01, len(listOfRanks))
#             else: #Keep non-RA the same
#                 listOfRanks = np.linspace(5, 500, 50, dtype=int)
#                 listOfRanks = [x for x in listOfRanks]
#                 errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
#         else: 
#             listOfRanks = np.linspace(5, 500, 50, dtype=int)
#             if rankAdapt:
#                 listOfRanks = [5 for _ in listOfRanks]
#                 errTols = genEvenlySpacedLogPoints(1.0, 0.00001, len(listOfRanks))
#             else: #Keep non-RA the same
#                 listOfRanks = [x for x in listOfRanks]
#                 errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
#     if decayType == "mid":
#         if samplingFactor == 1.0: #non-PS
#             if rankAdapt: #keep RA non-PS the same
#                 listOfRanks = np.linspace(5, 500, 50, dtype=int)
#                 listOfRanks = [5 for _ in listOfRanks] 
#                 errTols = genEvenlySpacedLogPoints(1.0, 0.01, len(listOfRanks))
#             else: #Keep non-RA the same
#                 listOfRanks = np.linspace(5, 500, 50, dtype=int)
#                 listOfRanks = [x for x in listOfRanks]
#                 errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
#         else: 
#             listOfRanks = np.linspace(5, 500, 50, dtype=int)
#             if rankAdapt:
#                 listOfRanks = [5 for _ in listOfRanks]
#                 errTols = genEvenlySpacedLogPoints(1.0, 0.00001, len(listOfRanks))
#             else: #Keep non-RA the same
#                 listOfRanks = [x for x in listOfRanks]
#                 errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
#     if decayType == "bot":
#         if samplingFactor == 1.0: #non-PS
#             if rankAdapt: #keep RA non-PS the same
#                 listOfRanks = np.linspace(5, 500, 50, dtype=int)
#                 listOfRanks = [5 for _ in listOfRanks] 
#                 errTols = genEvenlySpacedLogPoints(1.0, 0.00001, len(listOfRanks))
#             else: #Keep non-RA the same
#                 listOfRanks = np.linspace(5, 500, 50, dtype=int)
#                 listOfRanks = [x for x in listOfRanks]
#                 errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
#         else: 
#             listOfRanks = np.linspace(5, 500, 50, dtype=int)
#             if rankAdapt:
#                 listOfRanks = [5 for _ in listOfRanks]
#                 errTols = genEvenlySpacedLogPoints(1.0, 0.00001, len(listOfRanks))
#             else: #Keep non-RA the same
#                 listOfRanks = [x for x in listOfRanks]
#                 errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption

#TempAnalysis12
    # if decayType == "top":
    #     if samplingFactor == 1.0: #non-PS
    #         if rankAdapt: #keep RA non-PS the same
    #             listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #             listOfRanks = [5 for _ in listOfRanks] 
    #             errTols = genEvenlySpacedLogPoints(1.0, 0.01, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    #     else: 
    #         listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #         if rankAdapt:
    #             listOfRanks = [5 for _ in listOfRanks]
    #             errTols = genEvenlySpacedLogPoints(1.0, 0.00001, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    # if decayType == "mid":
    #     if samplingFactor == 1.0: #non-PS
    #         if rankAdapt: #keep RA non-PS the same
    #             listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #             listOfRanks = [5 for _ in listOfRanks] 
    #             errTols = genEvenlySpacedLogPoints(1.0, 0.01, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    #     else: 
    #         listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #         if rankAdapt:
    #             listOfRanks = [5 for _ in listOfRanks]
    #             errTols = genEvenlySpacedLogPoints(1.0, 0.00001, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    # if decayType == "bot":
    #     if samplingFactor == 1.0: #non-PS
    #         if rankAdapt: #keep RA non-PS the same
    #             listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #             listOfRanks = [5 for _ in listOfRanks] 
    #             errTols = genEvenlySpacedLogPoints(1.0, 0.001, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    #     else: 
    #         listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #         if rankAdapt:
    #             listOfRanks = [5 for _ in listOfRanks]
    #             errTols = genEvenlySpacedLogPoints(1.0, 0.00001, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption

    #New
    # if decayType == "top":
    #     if samplingFactor == 1.0: #non-PS
    #         if rankAdapt: #keep RA non-PS the same
    #             listOfRanks = np.linspace(5, 990, 50, dtype=int)
    #             listOfRanks = [5 for _ in listOfRanks] 
    #             errTols = genEvenlySpacedLogPoints(1.0, 0.1, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = np.linspace(5, 990, 50, dtype=int)
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    #     else: 
    #         listOfRanks = np.linspace(5, 990, 50, dtype=int)
    #         if rankAdapt:
    #             listOfRanks = [5 for _ in listOfRanks]
    #             errTols = genEvenlySpacedLogPoints(1.0, 0.000001, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    # if decayType == "mid":
    #     if samplingFactor == 1.0: #non-PS
    #         if rankAdapt: #keep RA non-PS the same
    #             listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #             listOfRanks = [5 for _ in listOfRanks] 
    #             errTols = genEvenlySpacedLogPoints(1.0, 0.01, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    #     else: 
    #         listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #         if rankAdapt:
    #             listOfRanks = [5 for _ in listOfRanks]
    #             errTols = genEvenlySpacedLogPoints(1.0, 0.000001, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    # if decayType == "bot":
    #     if samplingFactor == 1.0: #non-PS
    #         if rankAdapt: #keep RA non-PS the same
    #             listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #             listOfRanks = [5 for _ in listOfRanks] 
    #             errTols = genEvenlySpacedLogPoints(1.0, 0.00001, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    #     else: 
    #         listOfRanks = np.linspace(5, 500, 50, dtype=int)
    #         if rankAdapt:
    #             listOfRanks = [5 for _ in listOfRanks]
    #             errTols = genEvenlySpacedLogPoints(1.0, 0.00001, len(listOfRanks))
    #         else: #Keep non-RA the same
    #             listOfRanks = [x for x in listOfRanks]
    #             errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption


    #JOHN FINAL
    if decayType == "top":
        if samplingFactor == 1.0: #non-PS
            if rankAdapt: #keep RA non-PS the same
                listOfRanks = np.linspace(5, 500, 50, dtype=int)
                listOfRanks = [5 for _ in listOfRanks] 
                errTols = genEvenlySpacedLogPoints(1.0, 0.01, len(listOfRanks))
            else: #Keep non-RA the same
                listOfRanks = np.linspace(5, 500, 50, dtype=int)
                listOfRanks = [x for x in listOfRanks]
                errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
        else: 
            listOfRanks = np.linspace(5, 500, 50, dtype=int)
            if rankAdapt:
                listOfRanks = [5 for _ in listOfRanks]
                errTols = genEvenlySpacedLogPoints(1.0, 0.00001, len(listOfRanks))
            else: #Keep non-RA the same
                listOfRanks = [x for x in listOfRanks]
                errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    if decayType == "mid":
        if samplingFactor == 1.0: #non-PS
            if rankAdapt: #keep RA non-PS the same
                listOfRanks = np.linspace(5, 500, 50, dtype=int)
                listOfRanks = [5 for _ in listOfRanks] 
                errTols = genEvenlySpacedLogPoints(1.0, 0.01, len(listOfRanks))
            else: #Keep non-RA the same
                listOfRanks = np.linspace(5, 500, 50, dtype=int)
                listOfRanks = [x for x in listOfRanks]
                errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
        else: 
            listOfRanks = np.linspace(5, 500, 50, dtype=int)
            if rankAdapt:
                listOfRanks = [5 for _ in listOfRanks]
                errTols = genEvenlySpacedLogPoints(1.0, 0.00001, len(listOfRanks))
            else: #Keep non-RA the same
                listOfRanks = [x for x in listOfRanks]
                errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
    if decayType == "bot":
        if samplingFactor == 1.0: #non-PS
            if rankAdapt: #keep RA non-PS the same
                listOfRanks = np.linspace(5, 500, 50, dtype=int)
                listOfRanks = [5 for _ in listOfRanks] 
                errTols = genEvenlySpacedLogPoints(1.0, 0.001, len(listOfRanks))
            else: #Keep non-RA the same
                listOfRanks = np.linspace(5, 500, 50, dtype=int)
                listOfRanks = [x for x in listOfRanks]
                errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption
        else: 
            listOfRanks = np.linspace(5, 500, 50, dtype=int)
            if rankAdapt:
                listOfRanks = [5 for _ in listOfRanks]
                errTols = genEvenlySpacedLogPoints(1.0, 0.00001, len(listOfRanks))
            else: #Keep non-RA the same
                listOfRanks = [x for x in listOfRanks]
                errTols = [1.0 for _ in listOfRanks] #This won't be used since there is no rank adaption





    plotDict = {}
    for _ in range(10): 
        indices = list(range(len(listOfRanks)))
        random.shuffle(indices)
        for idx in indices:
            numComponents = listOfRanks[idx]
            rankAdaptMinError = errTols[idx]
            currFile = mypath+'/scalingTest_Time{}_NumComponents{}_ErrTol{}'.format(currRun, numComponents, str(rankAdaptMinError).replace('.', '-'))
            # writer = csv.writer(file)
            # writer.writerow(["Process Time", "Wall Time", "Reconstruction Error"])
            # keepArr = []

            freqDir = FreqDir(comm= 0, rank=0, size = 0, start_offset=0, num_imgs=0, exp=0, run=0,
                    det_type=det_type, output_dir=writeToHere, num_components=numComponents, alpha=alpha, rankAdapt=rankAdapt, rankAdaptMinError = rankAdaptMinError,
                    merger=False, mergerFeatures=0, downsample=downsample, bin_factor=bin_factor,
                    currRun = currRun, samplingFactor=samplingFactor, imgData = fullImgData, imgsTracked = imgsTracked, psi=0, usePSI=False)
            print("{} STARTING SKETCHING")
            st = time.process_time() 
            st1 = time.perf_counter()
            freqDir.run()
            et1 = time.perf_counter()
            print("FINISHED RUNNING")
            et = time.process_time() 
            perfTime = et1-st1
            print("PERFTIME:", perfTime)
            print("TESTING RECON ERROR:", fullImgData.T.shape, freqDir.sketch[:freqDir.ell, :].shape)
            # reconError = wrappingTest.lowMemoryReconstructionErrorScaled(wrappingTest.fullImgData.T, wrappingTest.matSketch[:wrappingTest.num_components, :]) #JOHN CHANGE 12/07/2024. For rank adaptive tests, this basically ruins the test since you restrict the number of components. 
            reconError = lowMemoryReconstructionErrorScaled(fullImgData.T, freqDir.sketch, freqDir.ell)
            print("FULL RECON ERROR: ", reconError)
            # field = [et-st, perfTime, reconError]
            # writer.writerow(field)
            # keepArr.append(field)

            # finProcessTime = sum([j[0] for j in keepArr])/len(keepArr)
            # finWallTime = sum([j[1] for j in keepArr])/len(keepArr)
            # finReconError = sum([j[2] for j in keepArr])/len(keepArr)
            # writer.writerow([])
            # writer.writerow([finProcessTime, finWallTime, finReconError])
            # writer.writerow([])
            
            # plotMe.append((finReconError, finProcessTime))
            plotDict.setdefault(currFile, []).append((reconError, et-st))

            # finalWrites.append([numComponents, sum(finProcessTime)/len(finProcessTime), sum(finWallTime)/len(finWallTime), sum(finReconError)/len(finReconError)])
            # finalWrites.append([numComponents, finProcessTime, finWallTime, finReconError])

            # writer.writerow([])
            # writer.writerow([finProcessTime])
            # writer.writerow([finWallTime])
            # writer.writerow([finReconError])
            # writer.writerow([])
            # writer.writerow(["det_type ="+ str(det_type),
            #     "grabImgSteps ="+ str(grabImgSteps),
            #     "writeToHere ="+ str(writeToHere),
            #     "start_offset="+str(start_offset), "num_imgs="+str(num_imgs),
            #     "num_components="+str(numComponents),
            #     "alpha="+str(alpha), "rankAdapt="+str(rankAdapt), 
            #     "rankAdaptMinError ="+ str(rankAdaptMinError),
            #     "downsample="+str(downsample), "bin_factor="+str(bin_factor), 
            #     "eluThreshold="+str(eluThreshold), "eluAlpha="+str(eluAlpha),
            #     "threshold="+str(threshold), "normalizeIntensity="+str(normalizeIntensity),
            #     "noZeroIntensity="+str(noZeroIntensity), "minIntensity ="+ str(minIntensity),
            #     "samplingFactor="+str(samplingFactor),
            #     "divBy="+str(divBy), "thresholdQuantile="+str(thresholdQuantile),
            #     "unitVar="+str(unitVar),
            #     "usePSI="+str(usePSI)])

    plotMe = []
    for _, dataList in plotDict.items():
        finProcessTime = sum([j[0] for j in dataList])/len(dataList)
        finReconError = sum([j[1] for j in dataList])/len(dataList)
        plotMe.append((finReconError, finProcessTime))
    np.save(tempFileName, np.array(plotMe))
    np.save(f'/sdf/home/w/winnicki/papertests_20240717/tempAnalysis/'+f"tempErrorVsTime_{decayType}_{rankAdapt}_{str(samplingFactor).replace('.', '-')}.npy", np.array(plotMe))

    # Determine the maximum number of tuples in any list to ensure the CSV has enough columns
    max_tuples = max(len(v) for v in plotDict.values())
    # Create a CSV file
    with open(mypath + '/full_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Create header
        header = ["Key"]
        for i in range(1, max_tuples + 1):
            header.extend([f"ProcessTime{i}_1", f"ReconError{i}_2"])
        writer.writerow(header)
        # Write data rows
        for key, tuples in plotDict.items():
            row = [key]
            for t in tuples:
                row.extend(t)
            # Fill in the missing values with empty strings if the list is shorter
            row.extend([""] * (max_tuples - len(tuples)) * 2)
            writer.writerow(row)

    # x = np.array([j[0] for j in finalWrites])

    # y = np.array([j[1] for j in finalWrites])
    # plt.scatter(x, y)
    # plt.savefig(mypath+'/figs/{}_processtime.png'.format(currRun))
    # plt.clf()

    # y = np.array([j[2] for j in finalWrites])
    # plt.scatter(x, y)
    # plt.savefig(mypath+'/figs/{}_walltime.png'.format(currRun))
    # plt.clf()

    # y = np.array([j[3] for j in finalWrites])
    # plt.scatter(x, y)
    # plt.savefig(mypath+'/figs/{}_reconerror.png'.format(currRun))
    # plt.clf()
    # with open(mypath+'/scalingTest_Time{}_FINAL_VALUES.csv'.format(currRun), 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Process Time", "Wall Time", "Reconstruction Error"])
    #     for row in finalWrites:
    #         writer.writerow(row)
    #     writer.writerow(["det_type ="+ str(det_type),
    #         "grabImgSteps ="+ str(grabImgSteps),
    #         "writeToHere ="+ str(writeToHere),
    #         "start_offset="+str(start_offset), "num_imgs="+str(num_imgs),
    #         "num_components="+str(numComponents),
    #         "alpha="+str(alpha), "rankAdapt="+str(rankAdapt), 
    #         "rankAdaptMinError ="+ str(rankAdaptMinError),
    #         "downsample="+str(downsample), "bin_factor="+str(bin_factor), 
    #         "eluThreshold="+str(eluThreshold), "eluAlpha="+str(eluAlpha),
    #         "threshold="+str(threshold), "normalizeIntensity="+str(normalizeIntensity),
    #         "noZeroIntensity="+str(noZeroIntensity), "minIntensity ="+ str(minIntensity),
    #         "samplingFactor="+str(samplingFactor),
    #         "divBy="+str(divBy), "thresholdQuantile="+str(thresholdQuantile),
    #         "usePSI="+str(usePSI)])

    print("FINISHED SCALING RUN")

if __name__ == "__main__":
    main()
