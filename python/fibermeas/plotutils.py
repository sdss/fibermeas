import numpy
import matplotlib.pyplot as plt
import seaborn as sns

from .constants import imgScale, ferrule2Top, betaArmWidth, MICRONS_PER_MM
from .constants import fiber2ferrule

def imshow(imgData):
    """Show an image

    Plots an image using matplotlib's pyplot.imshow, but adjusts the extent
    such that 0,0 is the lower left corner of the lower left pixel, and the
    origin is in the LL corner.

    Parameters
    ------------
    imgData : ndarray
        2D array representing an image [rows, columns].
    """
    nrows, ncols = imgData.shape
    # extent = [0, ncols, 0, nrows]
    plt.imshow(imgData, origin="lower") #, extent=extent)


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


def plotGridEval(df, filename):
    """Plot a vizualization of the grid search.  Save it to disk

    Parameters:
    -------------
    df : pandas.DataFrame
        dataframe loaded from "templateGridEval.csv"
    """
    plt.figure(figsize=(10,10))
    sns.lineplot(x="rot", y="maxCorr", hue="betaArmWidth", data=df)
    plt.savefig(filename, dpi=350)
    plt.close()


def plotSolnsOnImage(
    imgData, rot, betaArmWidth, ferruleCenRow,
    ferruleCenCol, fiberMeasDict, filename
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
    filename : str
        path to save file
    """
    plt.figure(figsize=(10,7))
    imshow(imgData)

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
            '--', markersize=3, color="w"
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

    # plt.plot([rightLine_plusy[0], rightLine_minusy[0]], [rightLine_plusy[1], rightLine_minusy[1]], '--', color="white", linewidth=0.5)
    # plt.plot([leftLine_plusy[0], leftLine_minusy[0]], [leftLine_plusy[1], leftLine_minusy[1]], '--', color="white", linewidth=0.5)
    # plt.plot([topLine_plusy[0], topLine_minusy[0]], [topLine_plusy[1], topLine_minusy[1]], '--', color="white", linewidth=0.5)


def plotMarginals(df):
    plt.figure()
    sns.lineplot(x="rot", y="maxCorr", hue="betaArmWidth", data=df)

    plt.figure()
    sns.lineplot(x="betaArmWidth", y="maxCorr", hue="rot", data=df)

    plt.figure()
    sns.scatterplot(x="rot", y="betaArmWidth", hue="maxCorr", data=df)

    # plt.figure()
    # sns.scatterplot(x="meanCol", y="meanRow", hue="maxCorr", data=df)


