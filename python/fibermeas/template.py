import itertools
import numpy
from skimage.filters import gaussian, sobel
from skimage.transform import rescale, rotate
from multiprocessing import Pool
import time
import os

from fibermeas import config

from .constants import met2top, imgScale, betaArmWidth
from .constants import baseDir, versionDir, templateDir, templateFilePath
from .constants import ferrule2Top, betaArmRadius

from . import __version__ as vers

# print("version", __version__)
# print("config", config)

MICRONS_PER_MM = 1000

# varied parameters for creating template grids
rotVary = numpy.linspace(-4, 4, 4)  # degrees
betaArmWidthVary = numpy.linspace(-0.1, .1, 4) + betaArmWidth  # mm
upsample = 1
blurMag = 1

# templateFilePath = os.path.join(templateDir, "templates.npy")


def betaArmTemplate(
    imgRot=0,
    betaArmWidth=betaArmWidth,
    upsample=upsample,
    blurMag=blurMag,
    imgScale=imgScale
):
    """
    Create a sorbel filtered outline of a beta arm, to be cross correlated
    with a real image.  Central pixel in template is the xy location of the
    nominal ferrule center.  So rotations of this image about its central
    pixel will be rotations about the ferrule center.

    Parameters
    -----------
    imgRot : float
        clockwise angle (deg) of beta arm, negative is CCW.
    betaArmWidth : float
        width of beta arm in mm
    upsample : int
        odd integer, upsample to get higher resolution (runs slower)
    blurMag : int
        gaussian blur magnitude in original pixel scale (not upsampled pix)
    imageScale : float
        image scale in microns per pixel

    Returns
    --------
    temp : numpy.ndarray
        the 2D image template of a beta arm outline (sorbel filtered)

    """

    if upsample % 2 == 0:
        raise RuntimeError("upsample parameter must be odd!")

    # pick a size for this template (big enough to get the whole beta arm in)
    size = int(2 * met2top * MICRONS_PER_MM / imgScale)

    if size % 2 == 0:
        size += 1  # make it odd so a pixel is centered
    temp = numpy.zeros((size * upsample, size * upsample))

    # midX/Y the central pixel in template
    # template is odd sized so this pixel is really the center
    midX = int(size * upsample / 2)
    midY = midX

    # draw beta arm outline
    # convert betaArmWidth to pixels
    betaArmWidth = betaArmWidth * MICRONS_PER_MM / imgScale * upsample
    # width must be odd to remain centered
    if betaArmWidth % 2 == 0:
        betaArmWidth += 1  # make it odd so a pixel is centered

    # this keeps the width centered on midX
    lside = midX - int(betaArmWidth / 2)
    rside = midX + int(betaArmWidth / 2) + 1

    temp[:, :lside] = 1
    temp[:, rside:] = 1

    topside = midY + int((ferrule2Top) * MICRONS_PER_MM / imgScale * upsample)
    temp[topside:, :] = 1

    curveRadPx = int(betaArmRadius * MICRONS_PER_MM / imgScale * upsample)
    yoff = topside - curveRadPx
    loff = lside + curveRadPx
    roff = rside - curveRadPx

    # left side
    columns = numpy.arange(lside,loff)
    rows = numpy.arange(yoff, topside)

    # all pixels in shoulder region
    larray = numpy.array(list(itertools.product(rows, columns)))

    for row, column in larray:
        dist = numpy.linalg.norm([row - yoff, column - loff])
        if dist > curveRadPx:
            temp[row, column] = 1

    # right side
    columns = numpy.arange(roff, rside)
    rows = numpy.arange(yoff, topside)

    # all pixels in shoulder region
    rarray = numpy.array(list(itertools.product(rows, columns)))

    for row, column in rarray:
        dist = numpy.linalg.norm([row - yoff, column - roff])
        if dist > curveRadPx:
            temp[row, column] = 1

    # apply sobel filter before rotating, this elimites extra edges
    # that pop up if you rotate before filtering
    # give it a bit of a blur by scale so it's approx blurMag pixels after downsampling
    temp = gaussian(temp, upsample * blurMag)
    temp = sobel(temp)

    # rotate whole image
    if imgRot != 0:
        temp = rotate(temp, imgRot)

    # scale back down to expected image size
    temp = rescale(temp, 1 / upsample, anti_aliasing=True)
    return temp


def multiTemplate(x):
    """Generate a template with specified (via index) rotation and beta
    arm width.  Intended for use with multiprocessing to generate a
    grid of templates.

    Parameters
    -----------
    x : tuple
        rotation index, beta arm with index

    Returns
    ---------
    temp : numpy.ndarray
        2D image template for a beta arm
    """
    i, j = x
    return betaArmTemplate(
        imgRot=rotVary[i],
        betaArmWidth=betaArmWidthVary[j],
        upsample=upsample
    )


def generateOuterTemplates():
    """Generate the grid of templates to be used for cross correlation.
    Save the (versioned) grid as a (big!) numpy file.  Ideally you only do this
    once (per version).
    """

    # check if grid exists already, if so don't do anything!
    if not os.path.exists(baseDir):
        print("creating fibermeas output directory")
        os.mkdir(baseDir)

    if not os.path.exists(versionDir):
        print("creating fibermeas version directory")
        os.mkdir(versionDir)

    if not os.path.exists(templateDir):
        print("creating fibermeas template directory")
        os.mkdir(templateDir)

    if os.path.exists(templateFilePath):
        print("template file already exists, no further action taken")
        return

    defaultImg = betaArmTemplate()
    templates = numpy.zeros((len(rotVary), len(betaArmWidthVary), defaultImg.shape[0], defaultImg.shape[1]))

    ijs = []
    for ii, imgRot in enumerate(rotVary):
        for jj, dBAW in enumerate(betaArmWidthVary):
            ijs.append([ii,jj])

    p = Pool(config["useCores"])
    t1 = time.time()
    templateList = p.map(multiTemplate, ijs)
    print("template gen took %.2f mins" % ((time.time() - t1) / 60.))
    p.close()

    for (i, j), temp in zip(ijs, templateList):
        templates[i,j,:,:] = temp
    numpy.save(templateFilePath, templates)

