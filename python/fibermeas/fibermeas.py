import os
import time
from multiprocessing import Pool

import numpy
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from skimage.filters import sobel
import fitsio

from fibermeas import config

from .constants import imgScale, ferrule2Top, betaArmWidth
from .constants import templateFilePath, versionDir
from .template import rotVary, betaArmWidthVary, upsample
from .plotutils import imshow

templates = numpy.load(templateFilePath)


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
    x : list
        index for rotation, index for beta arm width, image to correlate

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

    return rotVary[iRot], betaArmWidthVary[jBaw], maxResponse, argRow

def identifyFibers(imgData):
    """Measure the properties of backlit fibers in an image using
    skimage label/regionprops routines.

    Parameters
    ------------
    imgData : numpy.ndarray
        2D array of image data to measure

    Returns
    ---------
    regions : list skimage.RegionProperties
        Only things that look like they could be back lit fibers are returned.
    """

    # https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
    # might be useful to try?
    # https://stackoverflow.com/questions/31705355/how-to-detect-circlular-region-in-images-and-centre-it-with-python

    thresh = (data > numpy.percentile(data, 95))
    labels = label(thresh)
    props = regionprops(labels, data)

    regions = []
    for ii, region in enumerate(props):
        ed = region.equivalent_diameter
        if ed < 45 or ed > 60 or region.eccentricity > 0.3:
            # filter out unlikely regions by looking for circular
            # things of a certain size
            continue
        # cr, cc = region.weighted_centroid
        # plotCircle(cc+0.5, cr+0.5, ed/2)
        # plt.text(cc,cr,"%i"%ii)
        # print("%i"%ii, cr, cc, region.equivalent_diameter, region.bbox_area, region.eccentricity, region.perimeter)
        regions.append(region)

    return regions


def solveImage(imageFile):
    """Measure the locations of fibers in the beta arm reference frame.
    Measurement data and figures are put in a new directory:
    config["outputDirectory"]/imageFile/

    Parameters:
    ------------
    imageFile : str
        path to FITS file to analyze
    """
    imgName = imageFile.split("/")[-1].strip(".fits")
    imgName = imgName.strip(".FITS")

    measDir = os.path.join(
        versionDir,
        imgName
    )
    if os.path.exists(measDir):
        print("%s has already been processessed, skipping"%imageFile)
        return

    data = fitsio.read(imageFile)
    # normalize data
    data = data / numpy.max(data)
    # first identify fiber locations in the image
    fiberRegions = identifyFibers(data)

    # write fiber regions

    # correlate with beta arm templates to find origin of
    # beta arm coordinate system in the image frame
    indList = []
    for ii, imgRot in enumerate(rotVary):
        for jj, dBAW in enumerate(betaArmWidthVary):
            indList.append([ii,jj,data])

    pool = Pool(config["useCores"])
    t1 = time.time()
    out = numpy.array(pool.map(doOne, indList))
    print("took %2.f mins"%((t1-time.time())/60.))
    pool.close()

    maxCorr = pd.DataFrame(out, columns=["rot", "betaArmWidth", "maxCorr", "argRow", "argCol"])
    # write this as an output
    outFile = os.path.join(versionDir, "templateGridEval.csv")
    maxCorr.to_csv(outFile)

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


    # generateOuterTemplates()



    # refImg25sor = rotate(sobel(data), 2.5)

    # refImgsor = sobel(data)
    refImg = sobel(data)
    # fitsio.write("refImg.fits", refImg) # angle measured in ds9 90.15, 89.5, 359.9,

    # fits measured
    a = numpy.mean([90-90.15, 90-89.5, 90-(359.9-360+90)])
    print("a", a)

    refImg25 = rotate(refImg, 2.5)
    # fitsio.write("refImg25.fits", refImg25) # 87.688336, 87.288246, 357.56788
    a2 = numpy.mean([90-87.688336, 90-87.288246, 90-(357.56788-360+90)])
    print("a2", a2)


    # def genDataFrame():
    #     indList = []
    #     for ii, imgRot in enumerate(rotVary):
    #         for jj, dBAW in enumerate(betaArmWidthVary):
    #             indList.append([ii,jj,refImg25])

    #     pool = Pool(config["useCores"])
    #     t1 = time.time()
    #     out = numpy.array(pool.map(doOne, indList))
    #     print("took %2.f mins"%((t1-time.time())/60.))
    #     pool.close()
    #     maxCorr = pd.DataFrame(out, columns=["imgRot", "dBetaWidth", "maxCorr", "argRow", "argCol", "meanRow", "meanCol", "varRow", "varCol"])
    #     maxCorr.to_csv("maxCorr25.csv")



    # genDataFrame()

    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()

    # genDataFrame()

    # maxCorr is unrotated image
    # maxCorr25.csv is rotated by 2.5 degrees

    maxCorr = pd.read_csv("maxCorr.csv", index_col=0)

    plotMarginals(maxCorr)

    plt.show()



    maxCorr25 = pd.read_csv("maxCorr25.csv", index_col=0)

    plotMarginals(maxCorr25)

    plt.show()

    dbwDist = 0.02
    rotDist = 0.5

    sol, avgSol, maxCorrCut = findMaxResponse(maxCorr, dbwDist, rotDist)

    plotMarginals(maxCorrCut)

    plt.show()

    sol25, avgSol25, maxCorrCut25 = findMaxResponse(maxCorr25, dbwDist, rotDist)

    plotMarginals(maxCorrCut)
    plt.show()


    # amax = maxCorr["maxCorr"].idxmax()
    # sol = maxCorr.iloc[amax]

    # amax = maxCorr25["maxCorr"].idxmax()
    # sol25 = maxCorr25.iloc[amax]

    # import pdb; pdb.set_trace()

    plotResults([sol, avgSol], refImg)

    plotResults([sol25, avgSol25], refImg25)

    import pdb; pdb.set_trace()


