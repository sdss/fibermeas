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
from skimage.filters import sobel
from skimage.measure import regionprops, label
import fitsio
from jinja2 import Environment, PackageLoader, select_autoescape

from fibermeas import config

from .constants import imgScale, ferrule2Top, MICRONS_PER_MM
from .constants import betaAxis2ferrule, templateFilePath, versionDir, vers
from .constants import modelMetXY, modelBossXY, modelApXY
from .template import rotVary, betaArmWidthVary, upsample
from .plotutils import imshow, plotGridEval, plotSolnsOnImage, plotCircle

t1 = time.time()
templates = numpy.load(templateFilePath)
print("loading templates took %.1f seconds"%(time.time()-t1))


def computeAngles(centMeas):
    met = numpy.array([centMeas["metrology"]["centroidCol"], centMeas["metrology"]["centroidRow"]])
    ap = numpy.array([centMeas["apogee"]["centroidCol"], centMeas["apogee"]["centroidRow"]])
    bo = numpy.array([centMeas["boss"]["centroidCol"], centMeas["boss"]["centroidRow"]])

    dir1 = (bo - met)/numpy.linalg.norm(bo-met)
    dir2 = (ap - met)/numpy.linalg.norm(ap-met)

    ang = numpy.degrees(numpy.arccos(dir1.dot(dir2)))
    print("ang1", ang)

    dir1 = (met - bo)/numpy.linalg.norm(met-bo)
    dir2 = (ap - bo)/numpy.linalg.norm(ap-bo)

    ang = numpy.degrees(numpy.arccos(dir1.dot(dir2)))
    print("ang2", ang)

    dir1 = (met -ap)/numpy.linalg.norm(met-ap)
    dir2 = (bo -ap)/numpy.linalg.norm(bo-ap)

    ang = numpy.degrees(numpy.arccos(dir1.dot(dir2)))
    print("ang3", ang)


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

    plt.figure()
    imshow(image)

    plt.figure()
    imshow(template)


    plt.figure()
    imshow(corr)
    plotCircle(pixel[1], pixel[0], r=5)
    plt.show()

    return maxResponse, pixel


def doOne(x):
    """For use in multiprocessing.  Correlate an image with a template

    Parameters
    ------------
    x : list of [int, int, numpy.ndarray]
        [index for rotation, index for beta arm width, image to correlate]

    Returns
    ----------
    outputList : list
        1 or 2 element list, containing response for + and - rotation.
        Each list element contains 5 items:
            rot : float
                rotation of template
            betaArmWidth : float
                beta arm width of template
            maxResponse : float
                the maximum respose seen in the correlation
            row : int
                row in image where max response seen
            col : int
                column in image where max response seen
    """
    iRot = x[0]
    jBaw = x[1]
    refImg = x[2]

    # outputList = []

    rot = rotVary[iRot]
    betaArmWidth = betaArmWidthVary[jBaw]

    tempImg = templates[iRot,jBaw,:,:]
    maxResponse, [argRow, argCol] = correlateWithTemplate(refImg, tempImg)

    # outputList.append([rot, betaArmWidth, maxResponse, argRow, argCol])

    # now correlate negative rotation (flip image about y axis)
    # if rot != 0:
    #     tempImg = tempImg[:,::-1]
    #     maxResponse, [argRow, argCol] = correlateWithTemplate(refImg, tempImg)
    #     outputList.append([-1*rot, betaArmWidth, maxResponse, argRow, argCol])

    return rot, betaArmWidth, maxResponse, argRow, argCol


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


def processImage(imageFile, invertRow=False, invertCol=False):
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
    if invertRow:
        data = data[::-1,:]
    if invertCol:
        data = data[:,::-1]
    # normalize data
    data = data / numpy.max(data)
    # first identify fiber locations in the image

    # plt.figure()
    # imshow(sobel(data))
    # plt.show()

    fiberMeas = identifyFibers(data)

    # write fiber regions as a json file
    fiberMeasFile = os.path.join(measDir, "centroidMeas_%s_%s.json"%(imgName, vers))
    with open(fiberMeasFile, "w") as f:
        json.dump(fiberMeas, f, indent=4)

    # correlate with beta arm templates to find origin of
    # beta arm coordinate system in the image frame
    indList = []
    for ii, imgRot in enumerate(rotVary):
        for jj, dBAW in enumerate(betaArmWidthVary):
            indList.append([ii,jj,data])
            doOne(indList[-1])


    # pool = Pool(config["useCores"])
    # t1 = time.time()
    # out = numpy.array(pool.map(doOne, indList))
    # print("took %2.f mins"%((time.time()-t1)/60.))
    # pool.close()


    # flatten the output list to pack into a table
    # out = list(itertools.chain(*out))

    maxCorr = pd.DataFrame(out, columns=["rot", "betaArmWidth", "maxCorr", "argRow", "argCol"])
    # write this as an output
    outFile = os.path.join(measDir, "templateGridEval_%s_%s.csv"%(imgName, vers))
    maxCorr.to_csv(outFile)


def solveImage(imageFile, invertRow=False, invertCol=False):
    """Use the files output by processImage to determine the fiber coordinates
    in the beta arm coordinate system.  Various output files are written to:
    config["outputDirectory"]/imageFile/

    note: processImage must be run first!!! this routine expects outputs
    from that one.

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
    path2tempEval = os.path.join(measDir, "templateGridEval_%s_%s.csv"%(imgName, vers))
    tempEval = pd.read_csv(path2tempEval, index_col=0)
    tempEvalFigName = os.path.join(measDir, "templateGridEval_%s_%s.png"%(imgName, vers))
    plotGridEval(tempEval, tempEvalFigName)

    # find max response
    amax = tempEval["maxCorr"].idxmax() # where is the correlation maximized?
    argMaxSol = tempEval.iloc[amax]
    betaArmWidth = argMaxSol["betaArmWidth"]
    rot = argMaxSol["rot"]
    ferruleCenRow = argMaxSol["argRow"]
    ferruleCenCol = argMaxSol["argCol"]
    maxCorr = argMaxSol["maxCorr"]

    # overplot solutions on original image
    fiberMeasFile = os.path.join(measDir, "centroidMeas_%s_%s.json"%(imgName, vers))
    with open(fiberMeasFile, "r") as f:
        fiberMeasDict = json.load(f)

    # computeAngles(fiberMeasDict)

    imgData = fitsio.read(imageFile)
    if invertRow:
        imgData = imgData[::-1,:]
    if invertCol:
        imgData = imgData[:,::-1]

    fullFigName, zoomFigName = plotSolnsOnImage(
        imgData, rot, betaArmWidth, ferruleCenRow, ferruleCenCol,
        fiberMeasDict, measDir, imgName
    )

    # put centroid info in beta arm frame (mm)
    # unrotate (to put beta arm +x along image +y),
    # remember beta arm +x points from beta axis toward fibers,
    # along centerline of beta arm
    imgRot = numpy.radians(-1*rot)
    rotMat = numpy.array([
        [numpy.cos(imgRot), numpy.sin(imgRot)],
        [-numpy.sin(imgRot), numpy.cos(imgRot)]
    ])

    # center of rotation is the "nominal location" of the ferrule's center
    centRot = numpy.array([ferruleCenCol, ferruleCenRow])
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
    metErr = (modelMetXY - metXY)*MICRONS_PER_MM
    apErr = (modelApXY - apXY)*MICRONS_PER_MM
    boErr = (modelBossXY - boXY)*MICRONS_PER_MM
    # print("metErr", metErr, numpy.linalg.norm(metErr))
    # print("apErr", apErr, numpy.linalg.norm(apErr))
    # print("boErr", boErr, numpy.linalg.norm(boErr))

    # compile results
    solns = {}
    solns["ccdRot"] = rot  # angle (deg) from beta arm +x to ccd +y (CCW)
    solns["betaArmWidthMM"] = betaArmWidth  # best fit beta arm width (mm)
    # CCD col for expected center of ferrule
    # correlation returns the best pixel (not fractional!)
    # but error cannot be larger than 0.5 pixels or ~1 micron so
    # whatever
    solns["ferruleCenCCD"] = [ferruleCenCol, ferruleCenRow]
    solns["imgFile"] = imageFile
    solns["imgName"] = imgName
    solns["imgScale"] = imgScale  # microns per pixel
    solns["version"] = vers  # software version
    solns["maxCorr"] = maxCorr  # correlation strength
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
    solns["tempEvalFigName"] = tempEvalFigName
    solns["fullFigName"] = fullFigName
    solns["zoomFigName"] = zoomFigName

    # tbt
    solns["robotID"] = "N/A"
    solns["expTime"] = "N/A"
    solns["expDate"] = "N/A"
    solns["roboTailID"] = "N/A"

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


if __name__ == "__main__":
    pass

