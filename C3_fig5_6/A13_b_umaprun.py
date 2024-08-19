import sys
sys.path.append("/sdf/home/w/winnicki/btx/")
from btx.processing.freqdir import *

st = time.perf_counter()
skipSize = 1
#skipSize=1
#numImgsToUse = int(12000/skipSize)
numImgsToUse = -1

currRun = 20240521_065535
visMe = visualizeFD(
                # inputFile="/sdf/data/lcls/ds/mfx/mfxp23120/scratch/winnicki/h5writes_20240117/{}_ProjectedData".format(currRun),
                # inputFile=f"/sdf/data/lcls/ds/xpp/xppx22715/scratch/winnicki/h5writes_20240513/{currRun:04}_ProjectedData",
                inputFile=f"/sdf/data/lcls/ds/xpp/xppx22715/scratch/winnicki/h5writes_20240521/",
                outputFile=f"./UMAPVis_{currRun}.html",
                numImgsToUse=-1, #numImgsToUse=numImgsToUse,
                nprocs=1,
                userGroupings=[],
                includeABOD=True,
                skipSize = skipSize,
                # umap_n_neighbors=numImgsToUse//1500,
                umap_n_neighbors=10,
                umap_random_state=42,
                hdbscan_min_samples=int(480000*0.75//100),
#                hdbscan_min_samples=1,
                hdbscan_min_cluster_size=int(480000//100),
                optics_min_samples=20, optics_xi = 0.1, optics_min_cluster_size = 0.05,
                outlierQuantile=0.9)
visMe.fullVisualize()
visMe.userSave()
et = time.perf_counter()
print("UMAP HTML Generation Processing time: {}".format(et - st))
