import fitsio
from fibermeas.plotutils import imshow
import matplotlib.pyplot as plt
import numpy
from fibermeas.fibermeas import processImage, solveImage, templates
from fibermeas.plotutils import plotMarginals, imshow
from fibermeas.constants import versionDir

import pandas as pd
import matplotlib.pyplot as plt

from skimage.feature import canny
from skimage import exposure
from skimage.filters import rank
from skimage.morphology import disk
from skimage.morphology import ball
from skimage.filters import gaussian, sobel
import scipy
import seaborn as sns


def detectFirstFallingEdge(counts, pixels, thresh, skipPix=75):
    hitThresh = False
    countsMeas = []
    pixMeas = []

    for ii, (_counts, pix) in enumerate(zip(counts, pixels)):
        if ii < skipPix:
            continue # some artifacts at chip edge?
        if not hitThresh and _counts < -1*thresh:
            hitThresh = True
        if hitThresh:
            if _counts < -1*thresh:
                countsMeas.append(_counts)
                pixMeas.append(pix)
            else:
                break

    if len(pixMeas) < 2:
        return None

    pixMeas = numpy.array(pixMeas)
    countsMeas = numpy.array(countsMeas)

    # invert y's to make positive
    countsMeas *= -1

    # compute expected pixel value and variance
    pdf = countsMeas / numpy.sum(countsMeas)
    meanPix = numpy.sum(pdf * pixMeas)
    varPix = numpy.sum(pdf * pixMeas**2) - meanPix**2
    sn2 = numpy.max(countsMeas)/thresh
    skew = numpy.sum(pdf*(pixMeas-meanPix)**2)


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


if __name__ == "__main__":
    # imgName = "P1234_FTO00013_20201030.fits"
    imgName1 = "FPU_P9999_FTO88888.fits"
    imgName = "FPU_P0643_FTO12345.fits"

    imgData1 = fitsio.read(imgName)
    imgData1 = imgData1[::-1, :]
    imgData1 = imgData1[:, ::-1]

    # imgData1 = imgData1.T
    # imgData1 = imgData1[:500,:]
    imgData1 = gaussian(imgData1)

    imgData1 = imgData1 / numpy.max(imgData1)


    # plt.figure()
    # imshow(imgData1)

    # plt.figure()
    # imshow(imgData1.T[:, ::-1])
    # plt.show()


    # import pdb; pdb.set_trace()

    # imgData1 = gaussian(imgData1, 3)

    intDirs = ["r", "l", "d"]

    # intDirs = ["d"]
    xCent = []
    xVar = []
    yCent = []
    yVar = []
    allVar = []
    npts = []
    direction = []
    sn2 = []
    m2 = []

    for intDir in intDirs: # right, left, down
        # dont change if integrating rightward
        if intDir == "r":
            grad = numpy.gradient(imgData1, axis=1)
            grad, iqr = normalizeArr(grad)
            xPix = numpy.arange(grad.shape[-1]) + 0.5 # 0 equals 0.5 pixels
            for yPix, row in enumerate(grad):
                output = detectFirstFallingEdge(row, xPix, iqr*3)
                if output is None:
                    continue
                meanX, varX, nPts, _sn2, _m2 = output
                xCent.append(meanX)
                yCent.append(yPix + 0.5)
                xVar.append(varX)
                allVar.append(varX)
                yVar.append(0)
                direction.append(intDir)
                npts.append(nPts)
                sn2.append(_sn2)
                m2.append(_m2)

        elif intDir == "l":
            # reverse columns
            grad = numpy.gradient(imgData1[:, ::-1], axis=1)
            grad, iqr = normalizeArr(grad)
            xPix = numpy.arange(grad.shape[-1]) + 0.5 # 0 equals 0.5 pixels
            xPix = xPix[::-1]
            for yPix, row in enumerate(grad):
                output = detectFirstFallingEdge(row, xPix, iqr*3)
                if output is None:
                    continue
                meanX, varX, nPts, _sn2, _m2 = output
                xCent.append(meanX)
                yCent.append(yPix + 0.5)
                xVar.append(varX)
                allVar.append(varX)
                yVar.append(0)
                direction.append(intDir)
                npts.append(nPts)
                sn2.append(_sn2)
                m2.append(_m2)

        elif intDir == "d":
            # reverse columns
            grad = numpy.gradient(imgData1.T[:, ::-1], axis=1)
            grad, iqr = normalizeArr(grad)
            yPix = numpy.arange(grad.shape[-1]) + 0.5 # 0 equals 0.5 pixels
            yPix = yPix[::-1]
            for xPix, col in enumerate(grad):
                # print("xPix", xPix)
                output = detectFirstFallingEdge(col, yPix, iqr*3)
                if output is None:
                    continue
                meanY, varY, nPts, _sn2, _m2 = output
                xCent.append(xPix + 0.5)
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

    # estimate beta arm width
    # and for each CCD row
    baw = []
    bawVar = []
    yCent = []
    bawCenX = [] # x pixel center of beta arm
    bawCenXVar = [] # estimated variance of center location
    # estimate width
    yPixelsL = set(df[df.direction == "l"].yCent)
    yPixelsR = set(df[df.direction == "r"].yCent)
    yPixels = yPixelsL.intersection(yPixelsR)
    yPixels = numpy.array(list(yPixels))
    for yPixel in yPixels:
        lr = df[df.yCent == yPixel]
        r = lr[lr.direction == "r"]
        l = lr[lr.direction == "l"]
        _baw = r.xCent - l.xCent
        _bawVar = r.xVar + l.xVar
        _bawCenX = 0.5*(r.xCent + l.xCent)
        _bawCenXVar = 0.5**2(_bawVar)

        baw.append(_baw)
        bawVar.append(_bawVar)
        bawCenX.append(_bawCenX)
        bawCenXVar.append(_bawCenXVar)
        yCent.append(yPixel)

    betaMeasX = {}
    betaMeasX["baw"] = baw
    betaMeasX["bawVar"] = bawVar
    betaMeasX["yCent"] = yCent
    betaMeasX["bawCenX"] = bawCenX # x pixel center of beta arm
    betaMeasX["bawCenXVar"] = bawCenXVar # estimated variance of center location

    betaMeasX = pd.DataFrame(betaMeasX)

    plt.figure()
    sns.lineplot(x="yCent", y="baw", data=betaMeasX)
    plt.show()

    import pdb; pdb.set_trace()




    # # df = df[df.sn2 < 8]  # stronger detections are probably the wrong edge
    # # df = df[df.m2 < 12]  # get rid of very skewed pdfs
    # # df = df[(df.direction == "r") | (df.direction == "l")]
    # # df = df[df.npts >= 5]

    # plt.figure()
    # plt.hist(df.allVar, bins=100)
    # plt.title("allVar")

    # plt.figure()
    # plt.hist(df.npts, bins=100)
    # plt.title("npts")
    # # sns.histplot(data=df, x="allVar")
    # # plt.show()

    # plt.figure()
    # plt.hist(df.m2, bins=100)
    # plt.title("m2")

    # plt.figure()
    # plt.hist(df.sn2, bins=100)
    # plt.title("sn2")

    # plt.figure()
    # sns.scatterplot(x="xCent", y="yCent", hue="allVar", style="direction", data=df, alpha=0.5, edgecolors="none")
    # plt.axis("equal")
    # plt.show()

    # import pdb; pdb.set_trace()



    # grad = numpy.gradient(imgData1, axis=1)

    # # estimate background
    # bgCalc1 = list(imgData1[:, :350].flatten())
    # bgCalc2 = list(imgData1[:, 1410:].flatten())
    # bgCalc = numpy.array(bgCalc1+bgCalc2)
    # bgMean = numpy.mean(bgCalc)
    # bgStd = numpy.std(bgCalc)

    # imgData1 = imgData1 - bgMean



    # # imgData1 = numpy.gradient(imgData1,axis=1)

    # # imgData1 = imgData1.


    # # median = numpy.median(imgData1)

    # # cut = imgData1 < 0.2*median
    # # plt.figure()
    # # imshow(cut)
    # # plt.show()

    # # imgData1 = imgData1[200:400]
    # # imgData1 = sobel(imgData1)

    # plt.figure()
    # imshow(imgData1)


    # plt.figure()

    # meanOverY = numpy.mean(imgData1, axis=0)
    # detectFirstFallingEdge(meanOverY, bgStd*5)

    # yOff = 0
    # for ii, row in enumerate(imgData1):
    #     print("row", ii)
    #     detectFirstFallingEdge(row, bgStd*5)

        # xs = numpy.arange(len(row))
        # plt.plot(xs, row+yOff, color="black", alpha=0.01)
        # yOff += .01
        # break
    # plt.show()

    # xVals = numpy.arange(len(meanOverY))
    # # find location of first minima

    # hitThresh = False
    # minY = 9e9
    # minX = None
    # for xPix, yMean in enumerate(meanOverY):
    #     if not hitThresh and yMean < -1*bgStd*5:
    #         hitThresh = True
    #     if hitThresh:
    #         if yMean < minY:
    #             minY = yMean
    #             minX = xPix
    #         if yMean > minY:
    #             break

    # plt.figure()
    # plt.plot(xVals, meanOverY)
    # plt.plot(minX, minY, ".", color="red")
    # plt.show()




    # imgData1 = imgData1 > numpy.percentile(imgData1, 60)
    # plt.figure(figsize=(13,8))
    # plt.imshow(sobel(imgData1))
    # plt.title(imgName1)

    # # selem = disk(30)
    # # imgData1 = rank.equalize(imgData1, selem=selem)

    # imgData1 = exposure.equalize_hist(imgData1)
    # plt.figure(figsize=(13,8))
    # plt.imshow(sobel(imgData1))
    # plt.title(imgName1)

    # plt.show()

    # imgData = fitsio.read(imgName)
    # imgData = sobel(imgData)
    # # imgData = imgData > numpy.percentile(imgData, 60)
    # plt.figure(figsize=(13,8))
    # plt.imshow(imgData)
    # plt.title(imgName)
    # plt.show()


    # processImage(imgName, invertRow=True, invertCol=True)
    # solveImage(imgName, invertRow=True, invertCol=True)


    # dfFile = "/Users/csayres/fibermeas/v0.1/PU_P0643_FTO12345/templateGridEval_PU_P0643_FTO12345_0.1.csv"
    # tempEval = pd.read_csv(dfFile, index_col=0)
    # plotMarginals(tempEval)
    # plt.show()

    # d1 = fitsio.read(imgName)
    # d2 = fitsio.read(imgName2)
    # d2 = d2[::-1,:]

    # d1 = d1/numpy.max(d1)
    # d2 = d2/numpy.max(d2)

    # diff = d1-d2
    # plt.figure()
    # imshow(d1)
    # plt.figure()
    # imshow(d2)

    # plt.figure()
    # imshow(diff)
    # plt.show()