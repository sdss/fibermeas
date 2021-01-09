import numpy
import matplotlib.pyplot as plt
import seaborn as sns

from .constants import imgScale, ferrule2Top, betaArmWidth


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


def plotCircle(x, y, r, color="red"):
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
    plt.plot(xs, ys, color=color)


def plotResults(pdSeries, refImg):

    plt.figure()
    imshow(refImg)

    for pds in pdSeries:
        imgRot = numpy.radians(pds["imgRot"])
        rotMat = numpy.array([
            [numpy.cos(imgRot), numpy.sin(imgRot)],
            [-numpy.sin(imgRot), numpy.cos(imgRot)]
        ])

        dBW = pds["dBetaWidth"]
        meanRow = pds["meanRow"]
        meanCol = pds["meanCol"]
        # carefule with rows     / cols vs xy
        centerOffset = numpy.array([meanCol, meanRow])

        yHeight = 1.5*1000/imgScale

        # central axis of beta positioner
        x1,y1 = [0, ferrule2Top*1000/imgScale]
        x2,y2 = [0, -yHeight]

        # right edge of beta arm
        betaArmRad = (betaArmWidth + dBW)/2*1000/imgScale
        x3,y3 = [betaArmRad, ferrule2Top*1000/imgScale]
        x4,y4 = [betaArmRad, -yHeight]

        # left edge of beta arm
        x5,y5 = [-betaArmRad, ferrule2Top*1000/imgScale]
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

        plt.plot([midLine_plusy[0], midLine_minusy[0]], [midLine_plusy[1], midLine_minusy[1]], 'c', alpha=0.5, linewidth=0.5)
        plt.plot([rightLine_plusy[0], rightLine_minusy[0]], [rightLine_plusy[1], rightLine_minusy[1]], 'r', alpha=0.5, linewidth=0.5)
        plt.plot([leftLine_plusy[0], leftLine_minusy[0]], [leftLine_plusy[1], leftLine_minusy[1]], 'g', alpha=0.5, linewidth=0.5)
        plt.plot([topLine_plusy[0], topLine_minusy[0]], [topLine_plusy[1], topLine_minusy[1]], 'b', alpha=0.5, linewidth=0.5)
    plt.show()

def plotMarginals(df):
    plt.figure()
    sns.lineplot(x="imgRot", y="maxCorr", hue="dBetaWidth", data=df)

    plt.figure()
    sns.lineplot(x="dBetaWidth", y="maxCorr", hue="imgRot", data=df)

    plt.figure()
    sns.scatterplot(x="imgRot", y="dBetaWidth", hue="maxCorr", data=df)

    plt.figure()
    sns.scatterplot(x="meanCol", y="meanRow", hue="maxCorr", data=df)