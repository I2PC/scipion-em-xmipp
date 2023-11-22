import os

from pyworkflow.utils.path import cleanPath
from pwem.objects import SetOfCoordinates

from xmipp3.convert import readSetOfCoordinates


def readSetOfCoordsFromPosFnames(posDir, setOfInputCoords, sqliteOutName,
                                 write=True):

    """
      posDir: path where there are .pos files with coordinates
      setOfInputCoords. Set to find micrographs
      sqliteOutName. Path where sqlite map will be created. Warning, it overwrites
      content
    """

    inputMics = setOfInputCoords.getMicrographs()
    inputMicsPointer = setOfInputCoords.getMicrographs(asPointer=True)
    cleanPath(sqliteOutName)
    setOfOutputCoordinates= SetOfCoordinates(filename= sqliteOutName)
    setOfOutputCoordinates.setMicrographs(inputMicsPointer)
    setOfOutputCoordinates.setBoxSize(setOfInputCoords.getBoxSize())
    readSetOfCoordinates(posDir, micSet=inputMics,
                           coordSet=setOfOutputCoordinates,
                           readDiscarded=False)
    if write:
       setOfOutputCoordinates.write()
    return setOfOutputCoordinates


def writeCoordsListToPosFname(mic_fname, list_x_y, outputRoot):
    s = """# XMIPP_STAR_1 *
#
data_header
loop_
_pickingMicrographState
Auto
data_particles
loop_
_xcoor
_ycoor
"""
    baseName, _ = os.path.splitext(os.path.basename(mic_fname))
    print("%d %s %s"%(len(list_x_y), mic_fname,
                      os.path.join(outputRoot, baseName+".pos")))

    if len(list_x_y)>0:
        with open(os.path.join(outputRoot, baseName+".pos"), "w") as f:
            f.write(s)
            for x, y in list_x_y:
              f.write(" %d %d\n" % (x, y))


def writeCoordsListToRawText(mic_fname, list_x_y, outputRoot):
    baseName= os.path.basename(mic_fname).split(".")[0]
    with open(os.path.join(outputRoot, baseName+"_raw_coords.txt"), "w") as f:
        f.write("#%s %s\n"%(baseName, mic_fname))
        for x, y in list_x_y:
            f.write("%d %d\n"%(x, y))
