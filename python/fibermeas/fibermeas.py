import os
import time
from multiprocessing import Pool
import shutil
import json
import itertools
import datetime

import numpy
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from skimage.filters import sobel, gaussian
from skimage.measure import regionprops, label
import fitsio
import scipy
from jinja2 import Environment, PackageLoader, select_autoescape

from fibermeas import config

from . import constants
from .constants import ferrule2Top, MICRONS_PER_MM
from .constants import betaAxis2ferrule, versionDir, vers
from .constants import modelMetXY, modelBossXY, modelApXY
# from .template import rotVary, betaArmWidthVary, upsample
from .plotutils import imshow, plotSolnsOnImage, plotCircle

import seaborn as sns

# for use in edge detection
IQR_THRESH = 3  # throw out points beyond this inner quartile range
PX_SMOOTH = 5  # pixels to smooth the original image before edge detection


def detectFirstFallingEdge(counts, pixels, thresh, skipPix=75):
    hitThresh = False
    countsMeas = []
    pixMeas = []

    for ii, (_counts, pix) in enumerate(zip(counts, pixels)):
        if ii < skipPix:
            continue  # ignore the first few things
        if not hitThresh and _counts < -1 * thresh:
            hitThresh = True
        if hitThresh:
            if _counts < -1 * thresh:
                countsMeas.append(_counts)
                pixMeas.append(pix)
            else:
                break

    if len(pixMeas) < 2:
        # if less that two pixels form the bump this is garbage
        return None

    pixMeas = numpy.array(pixMeas)
    countsMeas = numpy.array(countsMeas)

    # invert "counts" to make positive, like a pdf should be
    # this is because the beta arm edge is a dip in intensity
    # rather than a bump
    countsMeas *= -1

    # compute expected pixel value and variance
    # an empirical pdf
    pdf = countsMeas / numpy.sum(countsMeas)
    meanPix = numpy.sum(pdf * pixMeas)
    varPix = numpy.sum(pdf * pixMeas**2) - meanPix**2
    sn2 = numpy.max(countsMeas) / thresh
    skew = numpy.sum(pdf * (pixMeas - meanPix)**2)

    # plt.figure()
    # plt.plot(pixels, counts)
    # plt.axvspan(meanPix-varPix, meanPix+varPix, color="red", alpha=0.2)
    # plt.axvline(meanPix, color="red")
    # plt.xlim([meanPix-200, meanPix+200])
    # plt.show()

    return meanPix, varPix, len(pixMeas), sn2, skew


def normalizeArr(a):
    median = numpy.median(a)
    iqr = scipy.stats.iqr(a)
    a = a - median
    return a, iqr

    # print("iqr")
    # bgCalc1 = list(imgData1[:, :350].flatten())
    # bgCalc2 = list(imgData1[:, 1410:].flatten())
    # bgCalc = numpy.array(bgCalc1+bgCalc2)
    # bgMean = numpy.mean(bgCalc)
    # bgStd = numpy.std(bgCalc)

    # imgData1 = imgData1 - bgMean


def detectBetaArmEdges(imgData):
    """
    Parameters
    ------------
    imgData : numpy.ndarray
        2D array of image data to measure

    Returns
    ---------
    edgeDetections : pd.DataFrame
        data frame containing edge detections of beta arm perimiter

    """
    imgData = gaussian(imgData, PX_SMOOTH)
    imgData = imgData / numpy.max(imgData)

    xCent = []  # ccd columns, discrete or empirically determined "expected value"
    xVar = []  # empirical variance, can have zeros, depending on scan direction
    yCent = []  # ccd rows, discrete or empirically determened "expected value"
    yVar = []  # emprical variance, can have zeros
    allVar = []  # can never have zeros, basically which axis is being measured
    npts = []  # number of pixels involved in the "gaussian" edge detection
    direction = []  # scan direction
    sn2 = []  # signal to noise, or something like it
    m2 = []   # second moment about mean

    # right-moving, left-moving, down moving **direction** (not side of arm!!!)
    # indicating which direction the gradient is taken
    # on the input image to detect the
    # left, right, and top edges
    intDirs = ["r", "l", "d"]

    for intDir in intDirs:

        if intDir == "r":
            # dont change if moving rightward
            grad = numpy.gradient(imgData, axis=1)
            grad, iqr = normalizeArr(grad)
            xPix = numpy.arange(grad.shape[-1])
            # xPix[0] = 0 center of first pixel
            for yPix, row in enumerate(grad):
                output = detectFirstFallingEdge(row, xPix, iqr * IQR_THRESH)
                if output is None:
                    continue
                meanX, varX, nPts, _sn2, _m2 = output
                xCent.append(meanX)
                yCent.append(yPix)
                xVar.append(varX)
                allVar.append(varX)
                yVar.append(0)
                direction.append(intDir)
                npts.append(nPts)
                sn2.append(_sn2)
                m2.append(_m2)

        elif intDir == "l":
            # reverse columns if moving leftward
            grad = numpy.gradient(imgData[:, ::-1], axis=1)
            grad, iqr = normalizeArr(grad)
            xPix = numpy.arange(grad.shape[-1])
            xPix = xPix[::-1]
            # xPix[-1] = 0 center of first pixel in image
            for yPix, row in enumerate(grad):
                output = detectFirstFallingEdge(row, xPix, iqr * IQR_THRESH)
                if output is None:
                    continue
                meanX, varX, nPts, _sn2, _m2 = output
                xCent.append(meanX)
                yCent.append(yPix)
                xVar.append(varX)
                allVar.append(varX)
                yVar.append(0)
                direction.append(intDir)
                npts.append(nPts)
                sn2.append(_sn2)
                m2.append(_m2)

        elif intDir == "d":
            # transpose and reverse columns if moving downward
            grad = numpy.gradient(imgData.T[:, ::-1], axis=1)
            grad, iqr = normalizeArr(grad)
            yPix = numpy.arange(grad.shape[-1])
            yPix = yPix[::-1]
            # xPix[-1] = 0 center of first pixel in image
            for xPix, col in enumerate(grad):
                # print("xPix", xPix)
                output = detectFirstFallingEdge(col, yPix, iqr * IQR_THRESH)
                if output is None:
                    continue
                meanY, varY, nPts, _sn2, _m2 = output
                xCent.append(xPix)
                yCent.append(meanY)
                xVar.append(0)
                yVar.append(varY)
                allVar.append(varY)
                direction.append(intDir)
                npts.append(nPts)
                sn2.append(_sn2)
                m2.append(_m2)

    dd = {}
    dd["xCent"] = xCent
    dd["yCent"] = yCent
    dd["xVar"] = xVar
    dd["yVar"] = yVar
    dd["npts"] = npts
    dd["direction"] = direction
    dd["allVar"] = allVar
    dd["sn2"] = sn2
    dd["m2"] = m2

    df = pd.DataFrame(dd)

    return df


def solveBetaFrameOrientation(edgeDetections, imgScale):
    """
    Given the detected edges of the beta arm, infer the
    orientation with respect to the CCD rows and columns

    Parameters
    ------------
    edgeDetections : pd.DataFrame
        output from `.detectBetaArmEdges`
    """
    df = edgeDetections

    # estimate beta arm width
    # and for each CCD row
    baw = []  # beta arm width
    bawVar = []  # variance of beta arm width
    yCent = []  # ccd row "centroid" of center line of beta arm
    bawCenX = []  # ccd column "centroid" of center line  of beta arm
    bawCenXVar = []  # estimated variance of ccd column of beta arm

    # find ccd rows with both l and r detections, ignore the rest
    yPixelsL = set(df[df.direction == "l"].yCent)
    yPixelsR = set(df[df.direction == "r"].yCent)
    yPixels = yPixelsL.intersection(yPixelsR)
    yPixels = numpy.array(list(yPixels))

    for yPixel in yPixels:
        # measure the ccd column center location of the beta arm
        # as a function of ccd row
        lr = df[df.yCent == yPixel]
        r = lr[lr.direction == "r"]
        l = lr[lr.direction == "l"]
        _baw = float(l.xCent) - float(r.xCent)
        _bawVar = float(r.xVar) + float(l.xVar)
        _bawCenX = 0.5 * (float(r.xCent) + float(l.xCent))
        _bawCenXVar = 0.5**2 * float(_bawVar)

        baw.append(_baw)
        bawVar.append(_bawVar)
        bawCenX.append(_bawCenX)
        bawCenXVar.append(_bawCenXVar)
        yCent.append(yPixel)

    betaMeasX = {}
    betaMeasX["baw"] = baw
    betaMeasX["bawVar"] = bawVar
    betaMeasX["yCent"] = yCent
    betaMeasX["bawCenX"] = bawCenX  # x pixel center of beta arm
    betaMeasX["bawCenXVar"] = bawCenXVar  # estimated variance of center location

    betaMeasX = pd.DataFrame(betaMeasX)

    plt.figure()
    sns.lineplot(x="yCent", y="baw", data=betaMeasX, color="blue")

    # keep data in the innerquartile range based on beta arm width

    plCut = numpy.percentile(betaMeasX.baw, 40) # effectively cuts out tip curve
    # puCut = numpy.percentile(betaMeasX.baw, 99.5)
    betaMeasXClip = betaMeasX[betaMeasX.baw > plCut]

    bawMeasMM = numpy.median(betaMeasXClip.baw) * imgScale / MICRONS_PER_MM

    # betaMeasX = betaMeasX[betaMeasX.baw < puCut]

    sns.lineplot(x="yCent", y="baw", data=betaMeasXClip, color="red")
    plt.xlabel("CCD row")
    plt.ylabel("beta arm width (pixel)")

    print("len", len(betaMeasXClip))

    # plt.figure()
    # sns.lineplot(x="yCent", y="baw", data=betaMeasXClip)

    # realWidths = numpy.array(betaMeasXClip.baw)*imgScale / 1000 # in mm
    # print("median width", numpy.median(realWidths), numpy.mean(realWidths))

    # minWidth = numpy.min(realWidths)
    # maxWidth = numpy.max(realWidths)
    # bins = numpy.arange(minWidth,maxWidth,0.005) # 5 micron spacing
    # plt.figure()
    # plt.hist(realWidths, bins=bins)
    # plt.title("baw")

    ctrLine = numpy.array(betaMeasXClip.bawCenX)
    ctrErr = numpy.sqrt(betaMeasXClip.bawCenXVar)
    ys = numpy.array(betaMeasXClip.yCent)

    # import pdb; pdb.set_trace()

    plt.figure()
    plt.errorbar(ctrLine, ys, xerr=ctrErr)
    # plt.ylim([0, imgData1.shape[1]])
    # plt.plot(ys, ctrLine, '.')
    # plt.xlim([200,600])
    # plt.show()

    # linear fit x as a function of y (chi2 min)
    # hogg paper data analysis recipes
    # Ax = b
    nPts = len(ctrLine)
    Cinv = numpy.diag(1 / ctrErr)
    Y = ctrLine
    A = numpy.ones((nPts,2))
    A[:, 1] = ys

    (xIntercept,xSlope), resid, rank, s = numpy.linalg.lstsq(A, Y)
    fitX = xIntercept + xSlope * ys

    plt.plot(fitX, ys, '--', color="red")
    plt.xlabel("CCD col")
    plt.ylabel("CCD row")
    plt.title("beta arm midline")

    # this takes into account sigmas for measurement
    # (b2,m2), resid, rank, s = numpy.linalg.lstsq(
    #     A.T @ Cinv @ A,
    #     A.T @ Cinv @Y
    # )

    # fitX2 = b2 + m2 * ys
    # plt.plot(ys, fitX2, '--', color="black")

    # import pdb; pdb.set_trace()
    # plt.show()

    # plot 2nd derivative
    # d1 = numpy.gradient(betaMeasX.baw)
    # d2 = numpy.gradient(d1)
    # plt.figure()
    # plt.plot(ys, d1)

    # print(b-b2, m/m2)

    # import pdb; pdb.set_trace()

    # project "down" measurements onto this line
    dfDown = df[df.direction == "d"]
    # move from y, x to x, y
    # yIntercept = -1*b/m
    # ySlope = 1/m

    # plt.figure()
    # sns.scatterplot(x="xCent", y="yCent", data=dfDown)
    # plt.show()

    # shift x data by xIntercept, such that
    # beta arm midline passes through origin
    xs = numpy.array(dfDown.xCent) - xIntercept
    ys = numpy.array(dfDown.yCent)

    xys = numpy.array([xs, ys])

    # rotate data normal to fit direction of beta arm midline
    ang = numpy.pi/2 - numpy.arctan2(1 / xSlope, 1)

    rotMat = numpy.array([
        [numpy.cos(ang), -numpy.sin(ang)],
        [numpy.sin(ang), numpy.cos(ang)]
    ])

    invRotMat = numpy.array([
        [numpy.cos(-ang), -numpy.sin(-ang)],
        [numpy.sin(-ang), numpy.cos(-ang)]
    ])

    xyRot = rotMat.dot(xys)

    # plt.figure()
    # plt.plot(xys[0,:], xys[1,:])
    # plt.title("no rot")

    # plt.figure()
    # plt.plot(xyRot[0,:], xyRot[1,:])
    # plt.title("Top Edge Detections")

    # find the "flat spot" on the beta arm tip
    nomRadMM = constants.betaArmWidth/2 - constants.betaArmRadius
    nomRadPx = nomRadMM * MICRONS_PER_MM / imgScale

    keep = numpy.abs(xyRot[0,:]) < nomRadPx
    xyRotClip = xyRot[:, keep]

    # plt.plot(xyRotClip[0, :], xyRotClip[1, :], color="red", label="used detections")
    # plt.xlabel("x value (rotated pixels)")
    # plt.ylabel("y value (rotated pixels)")
    # plt.legend()

    yArmTop = numpy.median(xyRotClip[1])
    yFerruleCen = yArmTop - ferrule2Top * MICRONS_PER_MM / imgScale
    xFerruleCen = 0

    # plt.figure()
    # plt.hist(xyRotClip[1], bins=15)
    # plt.axvline(yArmTop, color='red')
    # # plt.xlim([915, 930])
    # # plt.ylim([0,125])
    # plt.title("top edge detect\nnpts=%i"%len(xyRotClip[1]))
    # plt.xlabel("y value (rotated pixels)")

    # rotate back into ccdFrame
    ferruleCenCol, ferruleCenRow = invRotMat.dot([xFerruleCen, yFerruleCen])
    ferruleCenCol += xIntercept

    results = {}
    results["rot"] = numpy.degrees(ang)  # CW rotation of image wrt beta arm
    results["betaArmWidthMM"] = bawMeasMM  # mm
    results["ferruleCenRow"] = ferruleCenRow
    results["ferruleCenCol"] = ferruleCenCol
    results["xIntercept"] = xIntercept
    results["xSlope"] = xSlope

    return results, betaMeasX, betaMeasXClip, xyRot, xyRotClip


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

    thresh = (imgData > numpy.percentile(imgData, 95)) # was 95
    labels = label(thresh)
    props = regionprops(labels, imgData)

    regions = []
    for ii, region in enumerate(props):
        ed = region.equivalent_diameter
        p = region.perimeter
        fracPerim = p / (2*numpy.pi*ed/2)

        # compare the perimeter to find round things
        if fracPerim > 1.1:
            continue
        if fracPerim < 0.9:
            continue
        if ed < 35 or ed > 68 or region.eccentricity > 0.5:
            # filter out unlikely regions by looking for circular
            # things of a certain size
            continue
            # pass
            # continue
        # plotCircle(cc+0.5, cr+0.5, ed/2)
        # plt.text(cc,cr,"%i"%ii)
        # print("%i"%ii, cr, cc, region.equivalent_diameter, region.bbox_area, region.eccentricity, region.perimeter)
        regions.append(region)

    # plt.figure()
    # imshow(imgData)
    # for region in regions:
    #     yCent, xCent = region.weighted_centroid
    #     rad = region.equivalent_diameter/2
    #     print("equiv dia", region.equivalent_diameter)
    #     print("eccen", region.eccentricity)
    #     print("perim", region.perimeter, region.perimeter/(2*numpy.pi*rad))
    #     print("axis lengths", region.minor_axis_length, region.major_axis_length)
    #     plotCircle(xCent, yCent, rad)
    # plt.show()


    if len(regions) != 3:
        raise RuntimeError("Found > 3 Fibers!!!")

    # sort regions by row (metrology fiber should be lowest column)
    regions.sort(key=lambda region: region.weighted_centroid[0])

    # pop off the metrology fiber
    metRegion = regions.pop(0)

    # sort remaining regions by column (to identify apogee vs boss fiber)
    # boss is right of apogee when looking from top of beta arm
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


def processImage(imageFile, invertRow=True, invertCol=True):
    """Measure the locations of fibers in the beta arm reference frame.
    Raw measurement data are put in a new directory:
    config["outputDirectory"]/imageFile/

    Parameters:
    ------------
    imageFile : str
        path to FITS file to analyze
    """
    imgBaseName = os.path.basename(imageFile)
    imgName = imgBaseName.split(".")[0]

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

    imgHeader = fitsio.read_header(imageFile)
    imgScale = imgHeader["PIXSCALE"]
    scaleErr = imgHeader["SCALEERR"]
    robotID = imgHeader["FPUNAME"]
    tailID = imgHeader["ROBOTAIL"]
    exptime = imgHeader["EXPTIME"]
    operator = imgHeader["OPERATOR"]
    date = imgHeader["DATE-OBS"]

    data = fitsio.read(imageFile)
    if invertRow:
        data = data[::-1, :]
    if invertCol:
        data = data[:, ::-1]
    # normalize data
    data = data / numpy.max(data)
    # first identify fiber locations in the image

    # plt.figure()
    # imshow(sobel(data))
    # plt.show()

    fiberMeasDict = identifyFibers(data)

    # write fiber regions as a json file
    fiberMeasFile = os.path.join(measDir, "centroidMeas_%s_%s.json"%(imgName, vers))
    with open(fiberMeasFile, "w") as f:
        json.dump(fiberMeasDict, f, indent=4)


    # find edges and thus directions of the beta arm
    # coordinate system
    edgeDetections = detectBetaArmEdges(data)

    # write edge detections to file
    outFile = os.path.join(measDir, "edgeDetections_%s_%s.csv"%(imgName, vers))
    edgeDetections.to_csv(outFile)

    betaOrient = solveBetaFrameOrientation(edgeDetections, imgScale)

    results, betaMeasX, betaMeasXClip, xyRot, xyRotClip = betaOrient

    outFile = os.path.join(measDir, "results_%s_%s.csv"%(imgName, vers))
    with open(outFile, "w") as f:
        json.dump(results, f, indent=4)

    outFile = os.path.join(measDir, "betaMeasX_%s_%s.csv"%(imgName, vers))
    betaMeasX.to_csv(outFile)

    outFile = os.path.join(measDir, "betaMeasXClip_%s_%s.csv"%(imgName, vers))
    betaMeasXClip.to_csv(outFile)

    outFile = os.path.join(measDir, "xyRot_%s_%s.csv"%(imgName, vers))
    numpy.savetxt(outFile, xyRot)

    outFile = os.path.join(measDir, "xyRotClip_%s_%s.csv"%(imgName, vers))
    numpy.savetxt(outFile, xyRotClip)

    ##### generate plots #####

    imgList = plotSolnsOnImage(
        data, results["rot"], results["betaArmWidthMM"],
        results["ferruleCenRow"], results["ferruleCenCol"],
        fiberMeasDict, measDir, imgName, imgScale
    )

    imgList = list(imgList)


    # plot all edge detections
    figName = os.path.join(measDir, "allEdges_%s_%s.png"%(imgName, vers))
    plt.figure()
    xs = edgeDetections.xCent.to_numpy()
    ys = edgeDetections.yCent.to_numpy()
    plt.plot(xs, ys, '.k')
    plt.xlabel("CCD col")
    plt.ylabel("CCD row")
    plt.axis("equal")
    plt.title("All edge detections\nleft scan, right scan, down scan")
    plt.savefig(figName, dpi=350)
    imgList.append(figName)

    # plot beta arm width as a function of ccd y
    figName = os.path.join(measDir, "baw_%s_%s.png"%(imgName, vers))
    plt.figure()
    ys = betaMeasX.baw.to_numpy()
    xs = betaMeasX.yCent.to_numpy()
    plt.plot(xs, ys, '-k')
    plt.xlabel("CCD row")
    plt.ylabel("beta arm width (px)")

    ys = betaMeasXClip.baw.to_numpy()
    xs = betaMeasXClip.yCent.to_numpy()
    plt.plot(xs, ys, '-r', label="used (flat regime)")
    plt.legend()
    plt.title("L R edge separation")
    plt.savefig(figName, dpi=350)
    imgList.append(figName)

    # beta center line
    figName = os.path.join(measDir, "baCenterFit_%s_%s.png"%(imgName, vers))
    plt.figure()

    # plot beta center line

    ctrLine = numpy.array(betaMeasXClip.bawCenX)
    ctrErr = numpy.sqrt(betaMeasXClip.bawCenXVar)
    ys = numpy.array(betaMeasXClip.yCent)
    plt.errorbar(ctrLine, ys, xerr=ctrErr)

    fitX = results["xIntercept"] + results["xSlope"] * ys

    plt.plot(fitX, ys, '--', color="red", label="beta centerline fit")
    plt.xlabel("CCD col (px)")
    plt.ylabel("CCD row (px)")
    plt.title("beta arm midline")
    plt.legend()

    plt.savefig(figName, dpi=350)
    imgList.append(figName)


    # plot rotated top edge detections
    figName = os.path.join(measDir, "topEdge_%s_%s.png"%(imgName, vers))
    plt.figure()
    plt.plot(xyRot[0,:], xyRot[1,:], color="black")
    plt.axis("equal")
    plt.title("Top Edge Detections\n(down scan)")

    plt.plot(xyRotClip[0, :], xyRotClip[1, :], color="red", label="used (flat regime)")
    plt.xlabel("x (rotated pixels)")
    plt.ylabel("y (rotated pixels)")
    plt.legend()
    plt.savefig(figName, dpi=350)
    imgList.append(figName)

    figName = os.path.join(measDir, "topEdgeHist_%s_%s.png"%(imgName, vers))
    topEdge = numpy.median(xyRotClip[1])
    ndetect = len(xyRotClip[1])
    plt.figure()
    plt.hist(xyRotClip[1, :])
    plt.axvline(topEdge, color="red", label="median")
    plt.xlabel("y (rotated pixels)")
    plt.title("top edge of beta arm\n nPts=%i"%ndetect)
    plt.legend()
    plt.savefig(figName, dpi=350)
    imgList.append(figName)

    # plt.show()


    # solve for beta arm coordinates of fibers and summarize everything

    # put centroid info in beta arm frame (mm)
    # unrotate (to put beta arm +x along image +y),
    # remember beta arm +x points from beta axis toward fibers,
    # along centerline of beta arm
    imgRot = numpy.radians(-1* results["rot"])
    rotMat = numpy.array([
        [numpy.cos(imgRot), numpy.sin(imgRot)],
        [-numpy.sin(imgRot), numpy.cos(imgRot)]
    ])

    # center of rotation is the "nominal location" of the ferrule's center
    centRot = numpy.array([results["ferruleCenCol"], results["ferruleCenRow"]])

    ccdMetXY = [
        fiberMeasDict["metrology"]["centroidCol"],
        fiberMeasDict["metrology"]["centroidRow"]
    ]

    ccdApogeeXY = [
        fiberMeasDict["apogee"]["centroidCol"],
        fiberMeasDict["apogee"]["centroidRow"]
    ]

    ccdBossXY = [
        fiberMeasDict["boss"]["centroidCol"],
        fiberMeasDict["boss"]["centroidRow"]
    ]

    metXY = numpy.array(ccdMetXY) - centRot
    apXY = numpy.array(ccdApogeeXY) - centRot
    boXY = numpy.array(ccdBossXY) - centRot

    # print("raw xys")
    # print(metXY)
    # print(apXY)
    # print(boXY)

    # unrotate
    metXY = rotMat.dot(metXY)
    apXY = rotMat.dot(apXY)
    boXY = rotMat.dot(boXY)

    # print("rot xys")
    # print(metXY)
    # print(apXY)
    # print(boXY)

    # convert from pixels to mm
    _metXY = metXY*imgScale/MICRONS_PER_MM
    _apXY = apXY*imgScale/MICRONS_PER_MM
    _boXY = boXY*imgScale/MICRONS_PER_MM

    # in image beta arm +x is roughly image +y
    # rotate 90 degrees CW, beta frame puts +x from beta axis toward fibers
    # this could probably be handled earlier with the original rotation?
    # here just carefully exchange x and y's to make the 90 deg rotation
    metXY[0] = _metXY[1]
    metXY[1] = _metXY[0] * -1

    apXY[0] = _apXY[1]
    apXY[1] = _apXY[0] * -1

    boXY[0] = _boXY[1]
    boXY[1] = _boXY[0] * -1

    # translate origin from ferrule center to beta axis
    metXY[0] += betaAxis2ferrule
    apXY[0] += betaAxis2ferrule
    boXY[0] += betaAxis2ferrule

    # compute errors
    # vector points from measured location to expected
    # location in beta arm frame, units are microns
    metErr = (modelMetXY - metXY) * MICRONS_PER_MM
    apErr = (modelApXY - apXY) * MICRONS_PER_MM
    boErr = (modelBossXY - boXY) * MICRONS_PER_MM
    # print("metErr", metErr, numpy.linalg.norm(metErr))
    # print("apErr", apErr, numpy.linalg.norm(apErr))
    # print("boErr", boErr, numpy.linalg.norm(boErr))

    # compile results
    solns = {}
    solns["ccdRot"] = results["rot"]  # angle (deg) from beta arm +x to ccd +y (CCW)
    solns["betaArmWidthMM"] = results["betaArmWidthMM"]  # best fit beta arm width (mm)
    # CCD col for expected center of ferrule
    # correlation returns the best pixel (not fractional!)
    # but error cannot be larger than 0.5 pixels or ~1 micron so
    # whatever
    solns["ferruleCenCCD"] = [results["ferruleCenCol"], results["ferruleCenRow"]]
    solns["imgFile"] = imageFile
    solns["imgName"] = imgName
    solns["imgScale"] = imgScale  # microns per pixel
    solns["imgScaleErr"] = scaleErr
    solns["version"] = vers  # software version
    solns["measDate"] = datetime.datetime.now().isoformat()

    # centroids of pixels in original image
    solns["metrologyCenCCD"] = ccdMetXY
    solns["bossCenCCD"] = ccdBossXY
    solns["apogeeCenCCD"] = ccdApogeeXY

    # rough diameter in pixels in original image
    solns["metrologyDiaCCD"] = fiberMeasDict["metrology"]["equivalentDiameter"]
    solns["bossDiaCCD"] = fiberMeasDict["boss"]["equivalentDiameter"]
    solns["apogeeDiaCCD"] = fiberMeasDict["apogee"]["equivalentDiameter"]

    # rough eccentricity in pixels in original image
    solns["metrologyEccentricity"] = fiberMeasDict["metrology"]["eccentricity"]
    solns["bossEccentricity"] = fiberMeasDict["boss"]["eccentricity"]
    solns["apogeeEccentricity"] = fiberMeasDict["apogee"]["eccentricity"]

    # measurements in beta arm frame (mm)
    solns["metrologyXYmm"] = list(metXY)
    solns["bossXYmm"] = list(boXY)
    solns["apogeeXYmm"] = list(apXY)

    # errors in microns
    # vector points from measured location to expected
    # location in beta arm frame, units are microns
    solns["metrologyErrUm"] = list(metErr)
    solns["bossErrUm"] = list(boErr)
    solns["apogeeErrUm"] = list(apErr)

    solns["metrologyErrMagUm"] = numpy.linalg.norm(metErr)
    solns["bossErrMagUm"] = numpy.linalg.norm(boErr)
    solns["apogeeErrMagUm"] = numpy.linalg.norm(apErr)

    # paths to plots
    solns["imgList"] = imgList

    solns["robotID"] = robotID
    solns["expTime"] = exptime
    solns["expDate"] = date
    solns["roboTailID"] = tailID
    solns["operator"] = operator

    summaryFile = os.path.join(measDir, "summary_%s_%s.json"%(imgName, vers))
    with open(summaryFile, "w") as f:
        json.dump(solns, f, indent=4)

    # write html file
    env = Environment(
        loader=PackageLoader('fibermeas', 'htmlTemplates'),
        autoescape=select_autoescape(['html', 'xml'])
    )

    template = env.get_template("summary.html")
    output = template.render(solns)

    summaryPath = os.path.join(measDir, "summary_%s_%s.html"%(imgName, vers))
    with open(summaryPath, "w") as f:
        f.write(output)

