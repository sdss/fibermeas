import fitsio
from fibermeas.plotutils import imshow
import matplotlib.pyplot as plt
from fibermeas.fibermeas import processImage, solveImage, templates


if __name__ == "__main__":
    imgName = "P1234_FTO00013_20201030.fits"
    # processImage(imgName)
    solveImage(imgName)