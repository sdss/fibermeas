import os
import time
from multiprocessing import Pool
import shutil
import json

import numpy
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from skimage.filters import sobel
from skimage.measure import regionprops, label
import fitsio

from fibermeas import config

from .constants import imgScale, ferrule2Top, betaArmWidth
from .constants import templateFilePath, versionDir
from .template import rotVary, betaArmWidthVary, upsample
from .plotutils import imshow, plotGridEval, plotSolnsOnImage

t1 = time.time()
templates = numpy.load(templateFilePath)
print("loading templates took %.1f seconds"%(time.time()-t1))


def correlateWithTemplate(image, template):
    """Correlate an input image with a template.  Return the max response,
    and the pixel [row, column] in the image at which the max response is seen.
    The central pixel in the template corresponds to the ferrule center in
    the beta arm.

    Parameters
    -----------
    image : numpy.ndarray
        input raw image
    template : numpy.ndarray
        input template

    Returns
    ---------
    maxResponse : float
        The maximum value of the correlation
    pixel : tuple
        [row, column] in the image where max response is seen
        this pixel corresponds the the expected ferrule center in
        the image
    """

    image = sobel(image)
    corr = signal.fftconvolve(image, template[::-1,::-1], mode="same")
    maxResponse = numpy.max(corr)
    pixel = numpy.unravel_index(numpy.argmax(corr), corr.shape)
    return maxResponse, pixel


def doOne(x):
    """For use in multiprocessing.  Correlate an image with a template

    Parameters
    ------------
    x : list of [int, int, numpy.ndarray]
        [index for rotation, index for beta arm width, image to correlate]

    Returns
    ----------
    rot : float
        rotation of template
    betaArmWidth : float
        beta arm width of template
    maxResponse : float
        the maximum respose seen in the correlation
    row : int
        row in image where max response seen
    col : in
        column in image where max response seen
    """
    iRot = x[0]
    jBaw = x[1]
    refImg = x[2]

    tempImg = templates[iRot,jBaw,:,:]
    maxResponse, [argRow, argCol] = correlateWithTemplate(refImg, tempImg)

    return rotVary[iRot], betaArmWidthVary[jBaw], maxResponse, argRow, argCol


def identifyFibers(imgData):
    """Measure the properties of backlit fibers in an image using
    skimage label/regionprops routines.

    Parameters
    ------------
    imgData : numpy.ndarray
        2D array of image data to measure

    Returns
    ---------
    fiberDict : dict
        nested dictionary keyed by fiber [metrology, apogee, boss], containing
        measurements of fiber location, circularity, and diameter
    """

    # https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
    # might be useful to try?
    # https://stackoverflow.com/questions/31705355/how-to-detect-circlular-region-in-images-and-centre-it-with-python

    thresh = (imgData > numpy.percentile(imgData, 95))
    labels = label(thresh)
    props = regionprops(labels, imgData)

    regions = []
    for ii, region in enumerate(props):
        ed = region.equivalent_diameter
        if ed < 45 or ed > 60 or region.eccentricity > 0.3:
            # filter out unlikely regions by looking for circular
            # things of a certain size
            continue
        # plotCircle(cc+0.5, cr+0.5, ed/2)
        # plt.text(cc,cr,"%i"%ii)
        # print("%i"%ii, cr, cc, region.equivalent_diameter, region.bbox_area, region.eccentricity, region.perimeter)
        regions.append(region)

    if len(regions) != 3:
        raise RuntimeError("Found > 3 Fibers!!!")

    # sort regions by row (metrology fiber should be lowest column)
    regions.sort(key=lambda region: region.weighted_centroid[0])

    # pop off the metrology fiber
    metRegion = regions.pop(0)

    # sort remaining regions by column (to identify apogee vs boss fiber)
    # boss is righ of apogee when looking from top of beta arm
    regions.sort(key=lambda region: region.weighted_centroid[1])
    apRegion = regions[0]
    bossRegion = regions[1]
    fiberNames = ["metrology", "apogee", "boss"]
    regions = [metRegion, apRegion, bossRegion]

    fiberDict = {}
    for fiberName, region in zip(fiberNames, regions):
        cr, cc = region.weighted_centroid
        fiberDict[fiberName] = {
            "centroidRow": cr,
            "centroidCol": cc,
            "eccentricity": region.eccentricity,
            "equivalentDiameter": region.equivalent_diameter
        }

    return fiberDict


def processImage(imageFile):
    """Measure the locations of fibers in the beta arm reference frame.
    Raw measurement data are put in a new directory:
    config["outputDirectory"]/imageFile/

    The two files written are fiberMeas.json and templateGridEval.csv

    Parameters:
    ------------
    imageFile : str
        path to FITS file to analyze
    """
    imgBaseName = os.path.basename(imageFile)
    imgName = imgBaseName.strip(".fits").strip(".FITS")

    measDir = os.path.join(
        versionDir,
        imgName
    )

    if os.path.exists(measDir):
        print("%s has already been processessed, skipping"%imageFile)
        return
    else:
        os.mkdir(measDir)

    # copy image to the meas directory
    shutil.copy(imageFile, os.path.join(measDir, imgBaseName))

    data = fitsio.read(imageFile)
    # normalize data
    data = data / numpy.max(data)
    # first identify fiber locations in the image
    fiberMeas = identifyFibers(data)

    # write fiber regions as a json file
    fiberMeasFile = os.path.join(measDir, "fiberMeas.json")
    with open(fiberMeasFile, "w") as f:
        json.dump(fiberMeas, f, indent=4)

    # correlate with beta arm templates to find origin of
    # beta arm coordinate system in the image frame
    indList = []
    for ii, imgRot in enumerate(rotVary):
        for jj, dBAW in enumerate(betaArmWidthVary):
            indList.append([ii,jj,data])

    pool = Pool(config["useCores"])
    t1 = time.time()
    out = numpy.array(pool.map(doOne, indList))
    print("took %2.f mins"%((time.time()-t1)/60.))
    pool.close()

    maxCorr = pd.DataFrame(out, columns=["rot", "betaArmWidth", "maxCorr", "argRow", "argCol"])
    # write this as an output
    outFile = os.path.join(measDir, "templateGridEval.csv")
    maxCorr.to_csv(outFile)


def solveImage(imageFile):
    """Use the files output by processImage to determine the fiber coordinates
    in the beta arm coordinate system


    Parameters:
    ------------
    imageFile : str
        path to FITS file to analyze
    """
    imgBaseName = os.path.basename(imageFile)
    imgName = imgBaseName.strip(".fits").strip(".FITS")

    measDir = os.path.join(
        versionDir,
        imgName
    )

    # load the csv with template grid evaluations
    path2tempEval = os.path.join(measDir, "templateGridEval.csv")
    tempEval = pd.read_csv(path2tempEval, index_col=0)
    figname = os.path.join(measDir, "templateGridEval.png")
    plotGridEval(tempEval, figname)

    # find max response
    amax = tempEval["maxCorr"].idxmax() # where is the correlation maximized?
    argMaxSol = tempEval.iloc[amax]
    betaArmWidth = argMaxSol["betaArmWidth"]
    rot = argMaxSol["rot"]
    ferruleCenRow = argMaxSol["argRow"]
    ferruleCenCol = argMaxSol["argCol"]

    # overplot solutions on original image
    fiberMeasFile = os.path.join(measDir, "fiberMeas.json")
    with open(fiberMeasFile, "r") as f:
        fiberMeasDict = json.load(f)

    imgData = fitsio.read(imageFile)

    filename = "junk"

    plotSolnsOnImage(
        imgData, rot, betaArmWidth, ferruleCenRow,ferruleCenCol,
        fiberMeasDict, filename)

    plotSolnsOnImage(
        imgData, rot+.15, betaArmWidth, ferruleCenRow,ferruleCenCol,
        fiberMeasDict, filename)

    plt.show()

    # plt.show()

    # find the best correlated template

    # visualize results


def findMaxResponse(df, dbawDist, rotDist):
    """dbawDist is absoute deviation from dbaw at max
       rotDist is absoulte deviation from imgRot at max

       grab solutions in the locality of the max response

       returns
       argMaxSol: pandas series for parameters at maxCorrelation
       cutDF: sliced input dataframe with results falling within
            beta arm distance and rot distance constraints
    """
    amax = df["maxCorr"].idxmax() # where is the correlation maximized?
    argMaxSol = df.iloc[amax]
    dbaw = argMaxSol["dBetaWidth"]
    rot = argMaxSol["imgRot"]

    # search around the argmax to average
    df = df[numpy.abs(df["dBetaWidth"] - dbaw) <= dbawDist]
    cutDF = df[numpy.abs(df["imgRot"] - rot) <= rotDist].reset_index()

    # create an argMaxSol analog by averaging over nearby values
    avgMaxSol = {}
    for key in ["imgRot", "dBetaWidth", "argCol", "argRow", "meanCol", "meanRow"]:

        marg = cutDF.groupby([key]).sum()["maxCorr"]
        keyVal = marg.index.to_numpy()
        corrVal = marg.to_numpy()
        corrValNorm = corrVal / numpy.sum(corrVal)

        # determine expected value and variance
        meanPar = numpy.sum(keyVal*corrValNorm)
        varPar = numpy.sum((keyVal-meanPar)**2*corrValNorm)
        print("par stats", key, meanPar, varPar)


        avgMaxSol[key] = meanPar

        # plt.figure()
        # plt.plot(keyVal, corrValNorm, 'ok-')
        # plt.title(key)
        # plt.show()


        # import pdb; pdb.set_trace()



    return argMaxSol, pd.Series(avgMaxSol), cutDF


if __name__ == "__main__":
    pass


###################################

    # # generateOuterTemplates()

    # # refImg25sor = rotate(sobel(data), 2.5)

    # # refImgsor = sobel(data)
    # refImg = sobel(data)
    # # fitsio.write("refImg.fits", refImg) # angle measured in ds9 90.15, 89.5, 359.9,

    # # fits measured
    # a = numpy.mean([90-90.15, 90-89.5, 90-(359.9-360+90)])
    # print("a", a)

    # refImg25 = rotate(refImg, 2.5)
    # # fitsio.write("refImg25.fits", refImg25) # 87.688336, 87.288246, 357.56788
    # a2 = numpy.mean([90-87.688336, 90-87.288246, 90-(357.56788-360+90)])
    # print("a2", a2)


    # # def genDataFrame():
    # #     indList = []
    # #     for ii, imgRot in enumerate(rotVary):
    # #         for jj, dBAW in enumerate(betaArmWidthVary):
    # #             indList.append([ii,jj,refImg25])

    # #     pool = Pool(config["useCores"])
    # #     t1 = time.time()
    # #     out = numpy.array(pool.map(doOne, indList))
    # #     print("took %2.f mins"%((t1-time.time())/60.))
    # #     pool.close()
    # #     maxCorr = pd.DataFrame(out, columns=["imgRot", "dBetaWidth", "maxCorr", "argRow", "argCol", "meanRow", "meanCol", "varRow", "varCol"])
    # #     maxCorr.to_csv("maxCorr25.csv")



    # # genDataFrame()

    # # import pdb; pdb.set_trace()

    # # import pdb; pdb.set_trace()

    # # genDataFrame()

    # # maxCorr is unrotated image
    # # maxCorr25.csv is rotated by 2.5 degrees

    # maxCorr = pd.read_csv("maxCorr.csv", index_col=0)

    # plotMarginals(maxCorr)

    # plt.show()



    # maxCorr25 = pd.read_csv("maxCorr25.csv", index_col=0)

    # plotMarginals(maxCorr25)

    # plt.show()

    # dbwDist = 0.02
    # rotDist = 0.5

    # sol, avgSol, maxCorrCut = findMaxResponse(maxCorr, dbwDist, rotDist)

    # plotMarginals(maxCorrCut)

    # plt.show()

    # sol25, avgSol25, maxCorrCut25 = findMaxResponse(maxCorr25, dbwDist, rotDist)

    # plotMarginals(maxCorrCut)
    # plt.show()


    # # amax = maxCorr["maxCorr"].idxmax()
    # # sol = maxCorr.iloc[amax]

    # # amax = maxCorr25["maxCorr"].idxmax()
    # # sol25 = maxCorr25.iloc[amax]

    # # import pdb; pdb.set_trace()

    # plotResults([sol, avgSol], refImg)

    # plotResults([sol25, avgSol25], refImg25)

    # import pdb; pdb.set_trace()


