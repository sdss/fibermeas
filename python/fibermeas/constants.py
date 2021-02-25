import os

import numpy

from fibermeas import config
from . import __version__ as vers

# the following constants are derived
# from solid model measurements of the beta arm + ferrule assembly
# the edrawings model can be found within the model directory

# these values are used to construct templates to correlate
# with data taken from the microscope

# metrology xy position in solid model
# mm in xy beta arm frame
modelMetXY = numpy.array([14.314, 0])
# boss xy position in solid model
modelBossXY = numpy.array([14.965, -0.376])
# apogee xy position in solid model
modelApXY = numpy.array([14.965, 0.376])

MICRONS_PER_MM = 1000

# distance from center of metrology fiber to end of beta arm
met2top = 1.886  # mm

#distance from beta axis to end of beta arm
betaAxis2Top = 16.2  # mm

# linear distance between fiber centers
fiber2fiber = 0.751  # mm

# distance from fiber center to ferrule center
# (assuming all fibers lie at same radius from ferrule center)
# solve the equalateral triangle for it's center
fiber2ferrule = 0.5 * fiber2fiber / numpy.cos(numpy.radians(30))

# distance from center of ferrule to end of beta arm
# assumes metrology fiber is lies exactly on beta arm centerline
ferrule2Top = met2top - fiber2ferrule # mm

# distance from beta axis to ferrule center
betaAxis2ferrule = betaAxis2Top - ferrule2Top

# radius of curvature for beta arm corners
betaArmRadius = 1.2  # mm

# width of beta arm
betaArmWidth = 3  # mm

# fiber radius in microns
fiberCoreRad = 120/2.

# image scale in microns per pixel (from rough measure)
# imgScale = 2.78

# from ricks header microns per pixel
# imgScale = 3.3043


# directory stuff
baseDir = os.path.expanduser(config["outputDirectory"])
versionDir = os.path.join(baseDir, "v" + vers)
templateDir = os.path.join(versionDir, "templateGrid")
templateFilePath = os.path.join(templateDir, "templates.npy")
