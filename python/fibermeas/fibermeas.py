import numpy
import matplotlib.pyplot as plt
import fitsio
from skimage import feature, exposure
from skimage.morphology import erosion, white_tophat, black_tophat
from skimage.morphology import disk
from skimage.measure import regionprops, label
from fibermeas.plotutils import plotCircle
# from fibermeas.fibermeas import detectBetaArmEdges
import glob
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage.util import invert
from scipy.ndimage import gaussian_filter1d
from skimage.filters import gaussian, sobel, sobel_h, sobel_v, median
import seaborn as sns
from scipy.signal import find_peaks, peak_widths, peak_prominences
import pandas as pd
import seaborn as sns
import os
import shutil

MicronPerMM = 1000
IMG_SCALE_GUESS = 3.62


class BetaImgModel(object):
    betaAxis2Top = 16.2  # mm from beta axis to top of beta arm
    # metrology xy position in solid model
    # mm in xy beta arm frame
    modelMetXY = numpy.array([14.314, 0])
    # boss xy position in solid model
    modelBossXY = numpy.array([14.965, -0.376])
    # apogee xy position in solid model
    modelApXY = numpy.array([14.965, 0.376])

    def __init__(self, b, m, medianL, medianR, imgScale):
        """y = m*x + b is the equation of the top of beta arm in pixels
        medianL/R are the median x pixels of left and right images *after*
        rotatating and shifting the top of the beta arm to the x axis
        """
        self.b = b
        self.m = m
        self.imgScale = imgScale # micron per pixel
        self.medianL = medianL  # in rotated frame px
        self.medianR = medianR  # in rotated frame px
        # print("beta arm width mm", (medianR-medianL)*imgScale/1000.)
        self.midX = numpy.mean([medianL, medianR]) # in rotated frame px
        self.imgRotRad = numpy.arctan(m) # positive means CCD is rotated CW wrt beta arm
        self.imgRotDeg = numpy.degrees(self.imgRotRad)

        # go from image pixels to rotated
        # frame where top of beta arm is on the x axis
        self.img2rotMat = numpy.array([
            [numpy.cos(self.imgRotRad), numpy.sin(self.imgRotRad)],
            [-numpy.sin(self.imgRotRad), numpy.cos(self.imgRotRad)]
        ])

        self.rot2imgMat = numpy.array([
            [numpy.cos(-self.imgRotRad), numpy.sin(-self.imgRotRad)],
            [-numpy.sin(-self.imgRotRad), numpy.cos(-self.imgRotRad)]
        ])


    def img2rotFrame(self, xPix, yPix):
        """take xy pixels and rotate them into rotated frame
        where top line of beta arm is on the x axis
        returns xy pixels in rotated frame
        """
        yPix = yPix - self.b
        xy = numpy.array([xPix,yPix])
        xRot, yRot = self.img2rotMat @ xy
        return xRot, yRot

    def rot2imgFrame(self, xRot, yRot):
        """xRot yRot are rotated pixels the line y=0 is the top of the beta arm
        returns xPix, yPix for pixels in original image frame
        """
        xy = numpy.array([xRot, yRot])
        xPix, yPix = self.rot2imgMat @ xy
        yPix = yPix + self.b
        return xPix, yPix

    def rot2betaFrame(self, xRot, yRot):
        """beta frame is in mm, the x axis points along the axis of the
        beta arm, the apogee fiber is at +y, boss at -y, metrology at y=0
        x=0 is the axis of beta rotation.
        """
        # shift x values such that the midline of positioner
        # is aligned with the y axis at the origin
        xRot = xRot - self.midX

        # scale by image scale into mm
        xMM = xRot * self.imgScale / MicronPerMM
        yMM = yRot * self.imgScale / MicronPerMM

        # exchange the x and y axes (images of beta arm are 90 degrees rotation
        # from beta arm frame)
        xBeta = yMM
        yBeta = xMM
        # invert y axis
        yBeta = -1 * yBeta

        # finally shift all x points by the nominal distance from
        # beta axis to top of beta arm
        xBeta = xBeta + self.betaAxis2Top
        return xBeta, yBeta

    def beta2rotFrame(self, xBeta, yBeta):
        """the reverse procedure as above"""
        xBeta = xBeta - self.betaAxis2Top
        yBeta = -1 * yBeta

        xMM = yBeta
        yMM = xBeta

        xRot = xMM * MicronPerMM / self.imgScale
        yRot = yMM * MicronPerMM / self.imgScale

        xRot = xRot + self.midX
        return xRot, yRot

    def beta2imgFrame(self, xBeta, yBeta):
        xRot, yRot = self.beta2rotFrame(xBeta, yBeta)
        xPix, yPix = self.rot2imgFrame(xRot, yRot)
        return xPix, yPix

    def img2betaFrame(self, xPix, yPix):
        xRot, yRot = self.img2rotFrame(xPix, yPix)
        xBeta, yBeta = self.rot2betaFrame(xRot, yRot)
        return xBeta, yBeta


class MeasureImage(object):

    def __init__(self, litImageFileName, darkImageFileName, outputDir):
        # self.basename = litImageFileName.split("_")[1]  #PXXXXX
        junk, filename = os.path.split(litImageFileName)
        self.basename = filename.split("_")[1]
        basedir = os.path.dirname(litImageFileName)
        junk, self.basedir = os.path.split(basedir)
        self.litFile = litImageFileName
        self.darkFile = darkImageFileName
        self.outputDir = outputDir
        self.pltTitle = "%s %s"%(self.basedir, self.basename)

        # read and rotate images 180 degrees for my sanity
        self.litData = fitsio.read(litImageFileName)[::-1, ::-1]
        self.darkData = fitsio.read(darkImageFileName)[::-1, ::-1]
        self.litHeader = fitsio.read_header(litImageFileName)
        self.darkHeader = fitsio.read_header(darkImageFileName)
        # choose the "main" image for analysis
        # lit vs dark are mostly used for only centroid fibers
        # by difference imaging, but probably just wanna fit one
        # image
        # self.mainData = self.darkData
        # self.mainHeader = self.darkHeader
        self.mainData = self.litData
        self.mainHeader = self.litHeader

        # None's below populated by detectEdges method
        # dataframe with x,y pixels for edge detections from
        # vertical and horizontal scans, including data like prom (signal)
        self.edgeDetections = None
        # left column for bounding box based on marginalized
        # signal over rows (leftmost bump on average in image)
        self.ii1 = None
        # right column for bounding box based on marginalized
        # signal over rows (rightmost bump on average in image)
        self.ii1 = None
        # top row for bounding box based on marginalized
        # signal over columns (topmost bump on average image)
        self.jj1 = None
        # dataframe containing data for all outer edge detections
        self.edgeDetections = None


        # None's below populated by fitBetaFrame method
        self.topEdgeSelection = None
        self.leftEdgeSelection = None
        self.leftEdgeSelection = None
        self.imgModel = None

        # populated by findFibers
        self.centroids = None

        self.plotList = [] # keep track of which plots are made


    # def plotROIs(self):
    #     fitWidth = 0.4 # mm top left right
    #     fitHeight = 0.1 # mm

    #     # estimate scale from fiber diameters
    #     imgscale = 120 / numpy.mean(self.centroids.dia)
    #     # equation through the ap/boss fibers
    #     metXYPix = self.centroids[self.centroids.fiberID == "Metrology"][["col", "row"]].to_numpy()[0]
    #     met2TopPix = MicronPerMM * (BetaImgModel.betaAxis2Top - BetaImgModel.modelMetXY[0]) / imgscale
    #     # topMidYPix = metXYPix[1] + MicronPerMM * (BetaImgModel.betaAxis2Top - BetaImgModel.modelMetXY) / IMG_SCALE_GUESS
    #     # topLPix = metXYPix[0] - MicronPerMM * 0.5 * fitWidth / IMG_SCALE_GUESS
    #     # topRPix = metXYPix[0] + MicronPerMM * 0.5 * fitWidth / IMG_SCALE_GUESS
    #     # LL = numpy.array([topLPix, topMidYPix - MicronPerMM * 0.5 * fitHeight / IMG_SCALE_GUESS])
    #     # UR = numpy.array([topRPix, topMidYPix - MicronPerMM * 0.5 * fitHeight / IMG_SCALE_GUESS])

    #     plt.imshow(self.mainData, origin="lower")
    #     plt.plot(metXYPix[0], metXYPix[1]+met2TopPix, 'xr')
    #     plt.show()

        # print(metXY)



    def process(self):
        self.findFibers()
        self.detectEdges()
        self.fitBetaFrame()

        self.plotList.append(self.plotRaw())
        self.plotList.extend(self.plotGradMarginals())
        self.plotList.append(self.plotTopFits())
        self.plotList.append(self.plotLRfits())
        self.plotList.append(self.plotEdgeDetections())
        self.plotList.extend(self.plotFinalSolution())
        # print(out)

    def plotFinalSolution(self):
        fig = plt.figure(figsize=(13,8))
        fig.suptitle(self.pltTitle)
        plt.imshow(exposure.equalize_hist(self.mainData), origin="lower", cmap="bone")

        # plot outlines of beta frame, draw them in rotated frame
        midL = self.imgModel.medianL
        midR = self.imgModel.medianR
        midLR = numpy.mean([midL, midR])

        xtop = numpy.array([midL, midR])
        ytop = numpy.array([0, 0])

        xl = numpy.array([midL, midL])
        xm = numpy.array([midLR, midLR])
        xr = numpy.array([midR, midR])
        yl = numpy.array([0, -900])

        rulers = [
            [xtop, ytop],
            [xl, yl],
            [xm, yl],
            [xr, yl]
        ]

        for x, y in rulers:
            xr, yr = self.imgModel.rot2imgFrame(x,y)
            plt.plot(xr, yr, ":", color="tab:blue")

        # plot expected fiber locations
        fiberExpect = [
            self.imgModel.modelApXY,
            self.imgModel.modelMetXY,
            self.imgModel.modelBossXY
        ]

        fiberIDs = ["Apogee", "Metrology", "BOSS"]

        for fiber in fiberExpect:
            # expected location
            x, y = self.imgModel.beta2imgFrame(fiber[0], fiber[1])
            # plt.scatter(x,y, s=1.5, edgecolor="red", color="none")
            plt.plot(
                x, y, "o", ms=6, markerfacecolor="None",
                markeredgecolor='tab:red', markeredgewidth=2
            )

        for fiberID in fiberIDs:
            frow = self.centroids[self.centroids.fiberID == fiberID]
            plt.plot(frow.col, frow.row, "x", ms=6, markeredgecolor="tab:red")
            # import pdb; pdb.set_trace()
            plotCircle(
                float(frow.col), float(frow.row),
                float(frow.dia)/2, color="tab:red", linestyle=":")

        filenames = [os.path.join(self.outputDir, "%s_all.png"%self.basename)]
        plt.savefig(filenames[0], dpi=350)

        for fiberID in fiberIDs:
            frow = self.centroids[self.centroids.fiberID == fiberID]
            x = float(frow.col)
            y = float(frow.row)
            plt.xlim([x-40, x+40])
            plt.ylim([y-40, y+40])
            plt.title(fiberID)
            filename = os.path.join(
                self.outputDir, "%s_all_%s.png"%(self.basename, fiberID)
            )
            plt.savefig(filename, dpi=350)
            filenames.append(filename)

        return filenames

    def plotRaw(self):
        ldata = self.litData
        ddata = self.darkData
        nData = ldata-ddata

        # fitsio.write("diff.fits", nData)
        # fitsio.write("eq.fits", exposure.equalize_hist(nData))
        # fitsio.write("eq2.fits", exposure.equalize_hist(ddata))
        fig, axs = plt.subplots(2,3, figsize=(13,6))

        axs = axs.flatten()
        fig.suptitle(self.pltTitle)
        fig.suptitle(self.pltTitle)
        axs[0].imshow(ldata, origin="lower", cmap="bone")
        axs[0].set_ylabel("raw")
        axs[1].imshow(ddata, origin="lower", cmap="bone")
        axs[2].imshow(nData, origin="lower", cmap="bone")

        axs[3].imshow(exposure.equalize_hist(ldata), origin="lower", cmap="bone")
        axs[3].set_ylabel("hist scaled")
        axs[4].imshow(exposure.equalize_hist(ddata), origin="lower", cmap="bone")
        axs[5].imshow(exposure.equalize_hist(nData), origin="lower", cmap="bone")

        for ax in axs:
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
        filename = os.path.join(self.outputDir, "%s_raw.png"%self.basename)
        plt.tight_layout()
        plt.savefig(filename, dpi=450)
        plt.close()
        return filename

    def findFibers(self):
        ################### arbitrary choices ###############
        erosionSize = 4  # pixels
        thresh = 99  # percentile
        ###################################################

        ldata = self.litData
        ddata = self.darkData
        nData = ldata-ddata  # difference image for finding fibers

        selem = disk(erosionSize)
        threshImg = nData > numpy.percentile(nData, thresh)
        eroded = erosion(threshImg, selem)

        # plt.imshow(eroded, origin="lower")
        # plt.show()

        # find regions
        labels = label(eroded)
        props = regionprops(labels, ldata)#nData)

        centroid = []
        for prop in props:
            cr, cc = prop.weighted_centroid
            if prop.equivalent_diameter < 25:
                # fiber dia is at least 25 pixels
                continue
            out = [cc, cr, prop.equivalent_diameter, prop.eccentricity]
            # print(out)
            centroid.append(out)

        # import pdb; pdb.set_trace()

        assert len(centroid) == 3

        centroid = numpy.array(centroid)
        # sort by column (x)
        ii = numpy.argsort(centroid[:, 0])
        centroid = centroid[ii]
        fiberID = ["Apogee", "Metrology", "BOSS"]

        d = {}
        d["col"] = centroid[:, 0]
        d["row"] = centroid[:, 1]
        d["dia"] = centroid[:, 2]
        d["ecen"] = centroid[:,3]
        d["fiberID"] = fiberID

        self.centroids = pd.DataFrame(d)

    def detectEdges(self):
        """
        Default to non-illuminated image for edge detections
        """

        ############# arbitrary decisions (by hand inspection) ###############
        # how many pixels to smooth the original data
        smoothPx = 3
        # how many pixels to smooth the gradient
        gradientSmoothPx = 8
        # for determining baselines above which to find a detection, these are
        # from the edge of the chip inward, where it's just noise and such
        backgroundPxRange = 35
        # how many stds above mean background to declare a "detection"
        # for left right gradient
        LRstdDetect = 10
        # how many stds above mean background to declare a "detection" for top
        TstdDetect = 10
        # how many pixels to grow the left, right, and top edge detections for
        # narrowing line by line detections of arm edge, ignoring everything
        # outside
        bboxBufferPx = 10
        #####################################################################

        # data = exposure.equalize_hist(self.mainData)
        data = self.mainData
        # images are noisy, smooth them a bit
        # (and smooth them again after gradients)
        data = gaussian(data, smoothPx)

        _cols = []
        _rows = []
        _side = []
        _prom = []
        # prom is basically signal (promiminance above surroundings)
        # height of bump above minimum on each side
        _width = []
        # _gradData = []

        # for horizontal gradients only look at y range between fibers
        self.minRow = int(self.centroids[self.centroids.fiberID == "Metrology"].row)
        self.maxRow = int(self.centroids[self.centroids.fiberID == "BOSS"].row)
        self.minCol = int(self.centroids[self.centroids.fiberID == "Apogee"].col)
        self.maxCol = int(self.centroids[self.centroids.fiberID == "BOSS"].col)
        # import pdb; pdb.set_trace()
        # print(minRow, maxRow)
        # first determine bounding box ii1, ii2, jj1 are limits
        for axis in [1, 0]:
            grad = numpy.gradient(data, axis=axis)
            # grad = numpy.abs(grad)   # try abs after smoothing?
            grad = gaussian_filter1d(grad, gradientSmoothPx, axis=axis)

            # _gradData.append(grad)

            if axis == 0:
                # iterate over columns instead of rows
                # save before transposing
                self.vertGrad = grad
                grad = grad.T
            else:
                self.horizGrad = grad

            # print("axis", axis)
            # plt.imshow(grad, origin="lower")
            # plt.show()
            if axis == 1:
                # meanVal = numpy.mean(grad, axis=0)
                # stdVal = numpy.std(grad, axis=0)
                meanVal = numpy.mean(grad[self.minRow:self.maxRow, :], axis=0)
                stdVal = numpy.std(grad[self.minRow:self.maxRow, :], axis=0)
            else:
                # meanVal = numpy.mean(grad, axis=0)
                # stdVal = numpy.std(grad, axis=0)
                meanVal = numpy.mean(grad[self.minCol:self.maxCol, :], axis=0)
                stdVal = numpy.std(grad[self.minCol:self.maxCol, :], axis=0)

            if axis == 1:
                # rBG = numpy.mean(meanVal[:backgroundPxRange])
                rBG = 0
                rSD = numpy.mean(stdVal[:backgroundPxRange])

                # find leftmost very big positive bump (chamfer edge)
                aw = numpy.argwhere(meanVal > (rBG + 2.5 * LRstdDetect * rSD)).flatten()
                self.leftChamf = aw[0]  # left side region of interest

                # find righmost very big negative bump (chamfer edge)
                aw = numpy.argwhere(meanVal < -1 * (rBG + 2.5 * LRstdDetect * rSD)).flatten()
                self.rightChamf = aw[-1]  # left side region of interest

                # find left most edge of robot small negative bump
                # before the left chamfer edge
                # begin scanning until zero crossing

                # grad2 = numpy.gradient(meanVal)

                score = 1e9
                for col in numpy.arange(0, self.leftChamf)[::-1]:
                    v = meanVal[col]
                    if v > score:
                        ii1 = col
                        break
                    score = v
                    #     valuesIncreasing = True

                    # score = v
                    # if valuesIncreasing and v > -1 * (rBG + 0.25 * LRstdDetect * rSD):
                    #     ii1 = col
                    #     break

                # find right most edge of robot, small positive bump
                # after chamfer edge
                score = -1e9
                for col in numpy.arange(self.rightChamf, len(meanVal)):
                    v = meanVal[col]
                    if v < score:
                        ii2 = col
                        break
                    score = v
                    #     valuesDecreasing = True
                    # score = v
                    # if valuesDecreasing and v < (rBG + 0.25 * LRstdDetect * rSD):
                    #     ii2 = col
                    #     break

                # aw = numpy.argwhere(meanVal < -1 * (rBG + 0.5 * LRstdDetect * rSD)).flatten()
                # # import pdb; pdb.set_trace()
                # aw = aw[aw < self.leftChamf]
                # ii1 = aw[0]

                # # find right most edge of robot, small positive bump
                # # after chamfer edge
                # aw = numpy.argwhere(meanVal > (rBG + 0.5 * LRstdDetect * rSD)).flatten()
                # aw = aw[aw > self.rightChamf]
                # ii2 = aw[-1]

                # aw = numpy.argwhere(meanVal < -1 * (rBG + LRstdDetect * rSD)).flatten()
                # ii1 = aw[0]  # left side region of interest
                # aw = numpy.argwhere(meanVal > rBG + LRstdDetect * rSD).flatten()
                # ii2 = aw[-1]  # right side region of interest

                # left right limits in image where beta arm should be
                self.ii1 = ii1 # save incase wanted later
                self.ii2 = ii2 # save incase wanted later
                # rough scale based on 3mm wide beta arm
                self.roughScale = 3 * MicronPerMM / (ii2 - ii1)

            else:
                # top side background
                # tBG = numpy.mean(meanVal[-backgroundPxRange:])
                tBG = 0
                tSD = numpy.mean(stdVal[-backgroundPxRange:])
                aw = numpy.argwhere(meanVal > (tBG + TstdDetect * tSD)).flatten()

                # top limit in image where beta arm should be
                #  ii1, ii2, jj1 form the rough frame of the beta arm
                jj1 = aw[-1]  # y limit region of interest
                self.jj1 = jj1  # save incase wanted later


        # find row by row individual peaks
        # only look inside bounding box
        for axis in [1, 0]:
            ### repeating this junk from before
            grad = numpy.gradient(data, axis=axis)
            grad = numpy.abs(grad)  # this time abs the grad
            grad = gaussian_filter1d(grad, gradientSmoothPx, axis=axis)
            if axis == 1:
                # for row in range(jj1 + bboxBufferPx):  # only iterate up to top of robot
                for row in range(self.minRow, self.maxRow+30):
                    line = grad[row]

                    peaks = find_peaks(line)[0]
                    prom = peak_prominences(line, peaks)[0]
                    widths = peak_widths(line, peaks)[0]

                    # ignore peaks not inside bounding box
                    keep = (peaks > ii1 - bboxBufferPx) & (peaks < ii2 + bboxBufferPx)
                    peaks = peaks[keep]
                    if len(peaks) == 0:
                        continue
                    prom = prom[keep]
                    widths = widths[keep]

                    # find first bump (left side)
                    _cols.append(peaks[0])
                    _rows.append(row)
                    _side.append("left")
                    _prom.append(prom[0])
                    _width.append(widths[0])

                    # find last bump (right side)
                    _cols.append(peaks[-1])
                    _rows.append(row)
                    _side.append("right")
                    _prom.append(prom[-1])
                    _width.append(widths[-1])

            else:
                grad = grad.T
                # for col in range(ii1 - bboxBufferPx, ii2 + bboxBufferPx):
                for col in range(self.minCol, self.maxCol):
                    line = grad[col]
                    peaks = find_peaks(line)[0]
                    prom = peak_prominences(line, peaks)[0]
                    widths = peak_widths(line, peaks)[0]
                    # ignore any maxima outside the region of interest
                    keep = peaks < jj1 + bboxBufferPx
                    peaks = peaks[keep]
                    if len(peaks) == 0:
                        continue
                    prom = prom[keep]
                    widths = widths[keep]

                    # find last bump (top of arm)
                    _cols.append(col)
                    _rows.append(peaks[-1])
                    _side.append("top")
                    _prom.append(prom[-1])
                    _width.append(widths[-1])

        df = {}
        df["col"] = _cols
        df["row"] = _rows
        df["side"] = _side
        df["prom"] = _prom
        df["width"] = _width

        self.edgeDetections = pd.DataFrame(df)

    def plotGradMarginals(self):
        # makes two figures, one for each marginal

        # first plot gradient along rows
        grad = self.horizGrad
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6,8))
        fig.suptitle(self.pltTitle)
        ax1.imshow(grad, origin="lower")
        ax1.set_title("%s Left-Right Smoothed Gradient"%self.basename)
        for line in grad[self.minRow:self.maxRow]:
            ax2.plot(numpy.arange(len(line)), line, '-', color="black", alpha=0.02)
        meanVal = numpy.mean(grad, axis=0)
        ax2.plot(numpy.arange(len(meanVal)), meanVal, '-', color="white", alpha=0.8)
        ax2.set_xlim([0, len(meanVal)])
        ax2.axvline(self.ii1, color="black", linestyle=":", alpha=0.6)
        ax2.axvline(self.ii2, color="black", linestyle=":", alpha=0.6)
        filename1 = os.path.join(self.outputDir, self.basename+"_lr_grad.png")
        plt.savefig(filename1, dpi=350)
        plt.close()


        # next plot gradient along cols
        grad = self.vertGrad
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,4))
        fig.suptitle(self.pltTitle)
        ax1.imshow(grad, origin="lower")
        ax1.set_title("%s Up-Down Smoothed Gradient"%self.basename)
        for line in grad.T[self.minCol:self.maxCol]: # traspose to iterate over columns
            ax2.plot(line, numpy.arange(len(line)), '-', color="black", alpha=0.02)
        meanVal = numpy.mean(grad, axis=1)
        ax2.plot(meanVal, numpy.arange(len(meanVal)), '-', color="white", alpha=0.8)
        ax2.set_ylim([0, len(meanVal)])
        ax2.axhline(self.jj1, color="black", linestyle=":", alpha=0.6)
        filename2 = os.path.join(self.outputDir, self.basename+"_ud_grad.png")
        plt.savefig(filename2, dpi=350)
        plt.close()
        return [filename1, filename2]

    def plotEdgeDetections(self):
        fig = plt.figure(figsize=(13,8))
        fig.suptitle(self.pltTitle)
        plt.imshow(self.mainData, origin="lower", cmap="bone")
        ax = plt.gca()
        cb = ax.scatter(
            self.edgeDetections.col, self.edgeDetections.row,
            c=self.edgeDetections.prom, s=1, cmap="cool"
        )
        plt.colorbar(cb, ax=ax, orientation="horizontal")
        filename = os.path.join(self.outputDir, self.basename+"_edgedetections.png")
        plt.savefig(filename, dpi=350)
        plt.close()
        return filename

    def _fitTop(self):
        # fit the top edge detections with a line
        # returns b, m for the line y=m*x + b
        # saves the points used to fit the model in a new dataframe

        ############ arbitrary decisions ####################
        # the width about the median column of top edgedetections
        # to use for line fitting
        # 1.5 is found by inspection to get most of the top flat
        topWidthMicron = 0.6 * MicronPerMM # * 1.8
        topWidthPx = topWidthMicron / self.roughScale # self.mainHeader["PIXSCALE"]
        # prom threshold is the minimum signal to declare a valid detectoin
        promThresh = 0.0002
        ####################################################

        df = self.edgeDetections
        dfTop = df[df.side == "top"]

        # dfTop = dfTop[dfTop.prom > promThresh]
        # midCol = numpy.median(dfTop.col)
        # dfTop = dfTop[
        #     (dfTop.col > midCol - topWidthPx/2) & (dfTop.col < midCol + topWidthPx/2)
        # ]

        # find the linear fit
        X = numpy.ones((len(dfTop), 2))
        X[:,1] = dfTop.col.to_numpy()
        y = dfTop.row.to_numpy()
        coef = numpy.linalg.lstsq(X, y)[0]

        self.topEdgeSelection = dfTop

        # plt.figure()
        # plt.plot(dfTop.col, dfTop.row, '.k')
        # plt.show()

        return coef[0], coef[1]

    def _fitEdges(self, b, m):
        # find the left and right sizes of the beta arm after
        # rotating detections by the (CCD rotation) angle found
        # in _fitTop with parameters b and m (y=mx+b)

        ############ arbitrary decisions ################
        # pick a (rotated!) column range to use points
        # for finding the left and right edges
        # top is 1.3 mm below the top of the beta arm
        # bottom is 2*1.3 mm below the top of the beta arm
        topThresh = -1.3*MicronPerMM / self.roughScale # self.mainHeader["PIXSCALE"]
        bottomThresh = 2*topThresh
        ###############################################

        imgRotRad = numpy.arctan(m)

        rotMat = numpy.array([
            [numpy.cos(imgRotRad), numpy.sin(imgRotRad)],
            [-numpy.sin(imgRotRad), numpy.cos(imgRotRad)]
        ])


        df = self.edgeDetections
        # rotate left and right detections by imgRot
        # and plot histograms
        lEdge = df[df.side=="left"][["col", "row"]].to_numpy()
        # lColors = df[df.side=="left"]["prom"].to_numpy()
        rEdge = df[df.side=="right"][["col", "row"]].to_numpy()
        # rColors = df[df.side=="right"]["prom"].to_numpy()

        # put the top of the beta arm on the x-axis
        lEdge[:,1] = lEdge[:,1] - b
        rEdge[:, 1] = rEdge[:,1] - b

        lrotated = (rotMat @ lEdge.T).T
        rrotated = (rotMat @ rEdge.T).T

        # only use data below 1.3 and above 2*1.3 mm ( the radius of the curve)
        # lkeep1 = lrotated[:,1] < topThresh  # inspection
        # lkeep2 = lrotated[:,1] > bottomThresh  # inspection
        # lkeep = lkeep1 & lkeep2
        # lrotated = lrotated[lkeep]
        # lEdge = lEdge[lkeep]

        # only use data below 1.2 mm ( the radius of the curve)
        # 1.3 mm below the top of beta arm and above 2*1.3
        # get the left right edges in the zone near the fiber heads
        # rkeep1 = rrotated[:,1] < topThresh  # inspection
        # rkeep2 = rrotated[:,1] > bottomThresh  # inspection
        # rkeep = rkeep1 & rkeep2
        # rrotated = rrotated[rkeep]
        # rEdge = rEdge[rkeep]

        # find the midpoint between left and right axes
        # on the rotated column axis
        medianL = numpy.median(lrotated[:,0])
        medianR = numpy.median(rrotated[:,0])


        # save the selections of original points used
        ldetections = df[df.side=="left"]
        # self.leftEdgeSelection = ldetections[lkeep]
        self.leftEdgeSelection = ldetections

        rdetections = df[df.side=="right"]
        # self.rightEdgeSelection = rdetections[rkeep]
        self.rightEdgeSelection = rdetections

        # plt.figure()
        # plt.plot(lrotated[:,0], lrotated[:,1])
        # plt.axvline(medianL)

        # plt.plot(self.leftEdgeSelection.col, self.leftEdgeSelection.row)
        # plt.show()

        return medianL, medianR

    def plotTopFits(self):
        fig = plt.figure()
        fig.suptitle(self.pltTitle)
        dfTop = self.edgeDetections[self.edgeDetections.side=="top"]
        plt.plot(dfTop.col, dfTop.row, '.', color="tab:orange")
        plt.plot(self.topEdgeSelection.col, self.topEdgeSelection.row, '.', color="tab:blue", alpha=0.5)
        xs = numpy.linspace(numpy.min(self.topEdgeSelection.col)-150, numpy.max(self.topEdgeSelection.col)+150)
        ys = self.imgModel.b + self.imgModel.m*xs
        plt.plot(xs, ys, '--', color="black")
        plt.title("Top Edge \nImg Rot = %.5f deg"%self.imgModel.imgRotDeg)
        plt.xlabel("image row")
        plt.ylabel("image col")
        filename = os.path.join(self.outputDir, "%s_topFit.png"%self.basename)
        plt.savefig(filename, dpi=350)
        plt.close()
        return filename

    def fitBetaFrame(self):
        b, m = self._fitTop()
        medianL, medianR = self._fitEdges(b, m)
        self.imgModel = BetaImgModel(b, m, medianL, medianR, self.roughScale) #self.mainHeader["PIXSCALE"])

    def plotLRfits(self):
        lEdge = self.leftEdgeSelection[["col", "row"]].to_numpy()
        rEdge = self.rightEdgeSelection[["col", "row"]].to_numpy()

        lrotX, lrotY = self.imgModel.img2rotFrame(lEdge[:,0], lEdge[:,1])
        rrotX, rrotY = self.imgModel.img2rotFrame(rEdge[:,0], rEdge[:,1])

        lsorted = numpy.sort(numpy.hstack((lEdge[:,0], lrotX)))
        rsorted = numpy.sort(numpy.hstack((rEdge[:,0], rrotX)))
        leftBins = numpy.arange(lsorted[0], lsorted[-1])
        rightBins = numpy.arange(rsorted[0], rsorted[-1])

        fig, axs = plt.subplots(2,2,figsize=(8,9))
        fig.suptitle(self.pltTitle)
        ax1, ax2, ax3, ax4 = axs.flatten()
        ax1.plot(lrotX, lrotY, ".", color="tab:blue", markersize=2, label="rotated")
        ax1.plot(lEdge[:,0], lEdge[:,1], ".", color="tab:orange", alpha=0.4, markersize=2, label="not rotated")
        ax1.set_title("left edge")

        ax1.legend()


        ax2.plot(rrotX, rrotY, ".", color="tab:blue", markersize=2)
        ax2.plot(rEdge[:,0], rEdge[:,1], ".", color="tab:orange", alpha=0.4, markersize=2)
        ax2.set_title("right edge")

        # plt.legend()


        # fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
        ax3.hist(lrotX, color="tab:blue", bins=leftBins, alpha=0.5)
        ax3.hist(lEdge[:,0], color="tab:orange", bins=leftBins, alpha=0.5)
        ax3.axvline(self.imgModel.medianL, color="red")
        # ax3.set_xlim([numpy.min(leftBins), numpy.max(leftBins)])

        ax4.hist(rrotX, color="tab:blue", bins=rightBins, alpha=0.5)
        ax4.axvline(self.imgModel.medianR, color="red")
        ax4.hist(rEdge[:,0], color="tab:orange", bins=rightBins, alpha=0.5)

        filename = os.path.join(self.outputDir, "%s_lr_fits.png"%self.basename)
        plt.savefig(filename, dpi=350)
        plt.close()
        return filename

# def plotFullSoln(imgName, b, m, medianL, medianR):
#     ddata = fitsio.read(imgName)
#     ddata = ddata[::-1,::-1]

#     # determine image rotation
#     imgRotRad = numpy.arctan(m)

#     # use to rotate things back to image frame
#     rotMat = numpy.array([
#         [numpy.cos(-imgRotRad), numpy.sin(-imgRotRad)],
#         [-numpy.sin(-imgRotRad), numpy.cos(-imgRotRad)]
#     ])

#     midpointX = numpy.mean([medianL, medianR])
#     nPts = 100

#     # build xys in rotated frame first
#     topLineXs = numpy.linspace(medianL, medianR, nPts)
#     topLineYs = numpy.zeros(nPts)
#     topLineXYs = numpy.array([topLineXs, topLineYs])

#     _xs = numpy.ones(nPts)
#     vertExtent = 800 # pixels
#     vertLineYs = numpy.linspace(-vertExtent, 0, nPts) # plot 600 pixels below top line

#     leftLineXYs = numpy.array([_xs*medianL, vertLineYs])
#     midLineXYs = numpy.array([_xs*midpointX, vertLineYs])
#     rightLineXYs = numpy.array([_xs*medianR, vertLineYs])

#     fig, axs = plt.subplots(1,2, figsize=(10, 5))
#     images = [exposure.equalize_hist(ddata), sobel(ddata)]
#     for ax, data, color in zip(axs, images, ["white", "black"]):
#         ax.imshow(data, origin="lower") #, cmap="jet")

#         # begin plotting rules
#         for xy in [topLineXYs, leftLineXYs, midLineXYs, rightLineXYs]:
#             x, y = rotMat @ xy
#             y = y + b
#             ax.plot(x,y, ":", color=color)
#         ax.set_xlim([numpy.min(leftLineXYs[0]) - 40, numpy.max(rightLineXYs[0])+40])

if __name__ == "__main__":


    allDirs = glob.glob("P*")
    ons = glob.glob("/Users/csayres/fibermeas/forConor/Sloan/*ON.fits")
    ons.sort()
    offs = glob.glob("/Users/csayres/fibermeas/forConor/Sloan/*OFF.fits")
    offs.sort()
    for on, off in zip(ons,offs):
        # print(d)
        mi = MeasureImage(on, off, "./")
        mi.process()

        break


        # mi.findFibers()
        # mi.plotRaw()

        # mi.detectEdges()
        # mi.plotGradMarginals()
        # mi.plotEdgeDetections()
        # mi.fitBetaFrame()
        # mi.plotTopFits()
        # mi.plotLRfits()
        # print(out)
        # break
    # plt.show()

    # sort into directories
    # allLights = glob.glob("FPU*L.fits")
    # allLights.sort()
    # allDarks = glob.glob("FPU*N.fits")
    # allDarks.sort()
    # for l,d in zip(allLights, allDarks):
    #     fpu = l.split("_")[1]
    #     os.mkdir(fpu)
    #     shutil.copy(l, fpu + "/" + l)
    #     shutil.copy(d, fpu + "/" + d)

    # old style:
    #

    # allLights = glob.glob("FPU*L.fits")
    # allLights.sort()
    # allDarks = glob.glob("FPU*N.fits")
    # allDarks.sort()

    # for di, li in zip(allDarks, allLights):
    #     findFibers(li, di)
    #     b, m, medianLRot, medianRRot = findEdges(di)
    #     plotFullSoln(di, b, m, medianLRot, medianRRot)
    #     break
    #     # import pdb; pdb.set_trace()
    #     # print(di, li)
    # plt.show()


# plt.figure()
# plt.imshow(exposure.equalize_hist(nData), origin="lower")

# plt.show()