# from pytest import mark
import numpy
import matplotlib.pyplot as plt

from fibermeas import template
from fibermeas.plotutils import imshow

numpy.random.seed(0)


def test_template():
    # general excercise
    imgRots = numpy.linspace(-4,4,50)
    betaWidths = numpy.linspace(2.9,3.1,50)
    upsamples = [1,3]
    blurMags = [1,2,4]
    imgScales = [2.78, 2.8, 2.5]
    for i in range(3):
        # get 5 templates randomly
        r = numpy.random.choice(imgRots)
        b = numpy.random.choice(betaWidths)
        u = numpy.random.choice(upsamples)
        m = numpy.random.choice(blurMags)
        s = numpy.random.choice(imgScales)
        temp = template.betaArmTemplate(
            imgRot=r,
            betaArmWidth=b,
            upsample=u,
            blurMag=m,
            imgScale=s
        )
    assert True


if __name__ == "__main__":
    imgRots = [-4, 0, 4]
    for imgRot in imgRots:
        plt.figure()
        temp = template.betaArmTemplate(imgRot=imgRot, betaArmWidth=3.2)
        plt.title("imgRot=%i"%imgRot)
        imshow(temp)
        print("sum: ", numpy.sum(temp))

    # betaWidths = [2.5, 3, 3.5]
    # for betaWidth in betaWidths:
    #     plt.figure()
    #     temp = template.betaArmTemplate(betaArmWidth=betaWidth)
    #     plt.title("betaWidth=%.2f"%betaWidth)
    #     imshow(temp)

    # blurMags = [1,3,9]
    # for blurMag in blurMags:
    #     plt.figure()
    #     temp = template.betaArmTemplate(blurMag=blurMag)
    #     plt.title("blurMag=%i"%blurMag)
    #     imshow(temp)

    plt.show()