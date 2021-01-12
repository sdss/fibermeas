import fitsio

from fibermeas.fibermeas import processImage, solveImage


if __name__ == "__main__":
    imgName = "P1234_FTO00013_20201030.fits"
    # processImage(imgName)
    solveImage(imgName)