import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import os
from skimage.filters import sobel

from .constants import ferrule2Top, betaArmWidth, MICRONS_PER_MM
from .constants import fiber2ferrule, vers


def imshow(imgData, doExtent=False):
    """Show an image

    Plots an image using matplotlib's pyplot.imshow, but adjusts the extent
    such that 0,0 is the lower left corner of the lower left pixel, and the
    origin is in the LL corner.

    Parameters
    ------------
    imgData : ndarray
        2D array representing an image [rows, columns].
    """

    if doExtent:
        nrows, ncols = imgData.shape
        extent = [0, ncols, 0, nrows]
        handle = plt.imshow(imgData, origin="lower", extent=extent)
    else:
        handle = plt.imshow(imgData, origin="lower")
    return handle


def plotCircle(x, y, r, color="red", linestyle="-"):
    """Plot a circle centered at [x,y] of radius r

    Paramters
    ----------
    x : float
        x position of circle center
    y : float
        y position of circle center
    r : float
        radius of circle
    color : str
        color for the circle
    """
    thetas = numpy.linspace(0, 2 * numpy.pi, 200)
    xs = r * numpy.cos(thetas)
    ys = r * numpy.sin(thetas)
    xs = xs + x
    ys = ys + y
    plt.plot(xs, ys, linestyle, color=color)


def plotSolnsOnImage(
    imgData, rot, betaArmWidth, ferruleCenRow,
    ferruleCenCol, fiberMeasDict, measDir,
    imageName, imgScale
):
    """
    Visualize results plotted over the original image.  Save to filename

    Parameters
    -----------
    imgData : numpy.ndarray
        2D array of image data overwhich to plot results
    rot : float
        rotation (deg) of beta arm in frame (+ CW)
    betaArmWidth : float
        width of beta arm (mm)
    ferruleCenRow : int
        row in imgData corresponding to max response of correlation
    ferruleCenCol : int
        column in imgData corresponding to max response of correlation
    fiberMeasDict : dict
        dictionary containing measurements of fiber positions
    measDir : str
        directory in which to save results
    imageName : str
        name of image corresponding to imgData (stripped of .fits)
    imgScale : float
        scale in microns per pixel
    Returns
    --------
    fullFigName : str
        path to full figure
    zoomFigName : str
        path to fiber-zoomed figure
    """

    plt.figure(figsize=(10,7))
    imshowHandle = imshow(imgData, doExtent=False)

    # plot lines indicating the beta arm
    # coordinate system
    imgRot = numpy.radians(rot)
    rotMat = numpy.array([
        [numpy.cos(imgRot), numpy.sin(imgRot)],
        [-numpy.sin(imgRot), numpy.cos(imgRot)]
    ])

    # dBW = pds["dBetaWidth"]
    # meanRow = pds["meanRow"]
    # meanCol = pds["meanCol"]


    # carefule with rows     / cols vs xy
    centerOffset = numpy.array([ferruleCenCol, ferruleCenRow])

    yHeight = 1.5*MICRONS_PER_MM/imgScale

    # central axis of beta positioner
    x1,y1 = [0, ferrule2Top*MICRONS_PER_MM/imgScale]
    x2,y2 = [0, -yHeight]

    # right edge of beta arm
    betaArmRad = betaArmWidth/2*MICRONS_PER_MM/imgScale
    x3,y3 = [betaArmRad, ferrule2Top*MICRONS_PER_MM/imgScale]
    x4,y4 = [betaArmRad, -yHeight]

    # left edge of beta arm
    x5,y5 = [-betaArmRad, ferrule2Top*MICRONS_PER_MM/imgScale]
    x6,y6 = [-betaArmRad, -yHeight]

    # top edge of beta arm
    x7,y7 = x5,y5
    x8,y8 = x3,y3

    # rotate lines by imgRot
    midLine_plusy = rotMat.dot([x1,y1]) + centerOffset # origin of coord sys
    midLine_minusy = rotMat.dot([x2,y2]) + centerOffset
    rightLine_plusy = rotMat.dot([x3,y3]) + centerOffset
    rightLine_minusy = rotMat.dot([x4,y4]) + centerOffset
    leftLine_plusy = rotMat.dot([x5,y5]) + centerOffset
    leftLine_minusy = rotMat.dot([x6,y6]) + centerOffset
    topLine_plusy = rotMat.dot([x7,y7]) + centerOffset
    topLine_minusy = rotMat.dot([x8,y8]) + centerOffset

    lineList = [
        [midLine_plusy, midLine_minusy],
        [rightLine_plusy, rightLine_minusy],
        [leftLine_plusy, leftLine_minusy],
        [topLine_plusy, topLine_minusy],
    ]

    for line_plusy, line_minusy in lineList:
        plt.plot(
            [line_plusy[0], line_minusy[0]],
            [line_plusy[1], line_minusy[1]],
            ':', markersize=2, color="w"
        )

    # plot ferrule center
    plt.plot(ferruleCenCol, ferruleCenRow, "o", markersize=8, markerfacecolor="w", markeredgecolor="black")

    # plot expected positions of fibers red dots
    r = fiber2ferrule * MICRONS_PER_MM / imgScale
    for _rot in [-90, -90+120, -90+120*2]:
        xf = r * numpy.cos(numpy.radians(_rot))
        yf = r * numpy.sin(numpy.radians(_rot))
        (xf,yf) = rotMat.dot([xf,yf]) + centerOffset
        plt.plot(xf, yf, "o", markersize=4, markeredgecolor="red", fillstyle="none")

    # plot measured positions of fibers (red x's) and radii
    for fiberID, fiberMeas in fiberMeasDict.items():
        r = fiberMeas["equivalentDiameter"]/2
        x = fiberMeas["centroidCol"]
        y = fiberMeas["centroidRow"]
        plt.plot(x,y, "x", markersize=4, color="red")
        plotCircle(x,y,r, linestyle=":")

    # save full figure
    fullFigName = os.path.join(measDir, "fullSoln_%s_%s.png"%(imageName, vers))
    zoomFigName = os.path.join(measDir, "zoomSoln_%s_%s.png"%(imageName, vers))
    plt.savefig(fullFigName, dpi=350)

    plt.xlim([ferruleCenCol-200, ferruleCenCol+200])
    plt.ylim([ferruleCenRow-200, ferruleCenRow+134])

    plt.savefig(zoomFigName, dpi=350)

    # plot measured positions of fibers (red x's) and radii
    figNames = []
    for doSobel in [False, True]:
        if doSobel:
            imshowData = imshowHandle.get_array()
            s = sobel(imshowData)
            s = s > numpy.percentile(s, 65)
            s = sobel(s)
            imshowHandle.set_array(s)
        for fiberID, fiberMeas in fiberMeasDict.items():
            r = fiberMeas["equivalentDiameter"]/2
            x = fiberMeas["centroidCol"]
            y = fiberMeas["centroidRow"]
            plt.xlim([x - r * 1.3, x + r * 1.3])
            plt.ylim([y - r * 1.3, y + r * 1.3])
            plt.title(fiberID)
            if doSobel:
                figName = os.path.join(measDir, "s_%s_%s_%s.png"%(fiberID, imageName, vers))
            else:
                figName = os.path.join(measDir, "%s_%s_%s.png"%(fiberID, imageName, vers))
            plt.savefig(figName, dpi=350)
            figNames.append(figName)

    allFigNames = [fullFigName, zoomFigName] + figNames

    return allFigNames



