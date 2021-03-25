# **************************************************************************
# *
# * Authors:     David Maluenda (dmaluenda@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************


import xmipp3
from pwem.objects import SetOfCoordinates
from pwem.viewers import  EmPlotter,  CoordinatesObjectView
from pyworkflow.protocol.params import IntParam, LabelParam, FloatParam
from pyworkflow.utils import cleanPath, makePath
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer

from pwem import emlib
from xmipp3.convert import getXmippAttribute
from xmipp3.protocols.protocol_deep_micrograph_screen import XmippProtDeepMicrographScreen

class XmippDeepMicrographViewer(ProtocolViewer):
    """         
        Viewer for the 'Xmipp - deep Micrograph cleaner' protocols.\n
        Select those cooridantes with high (close to 1.0)
        'zScoreDeepLearning2' value and save them.
        The Histogram may help you to decide a threshold.
    """
    _label = 'viewer deep Micrograph cleaner'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [XmippProtDeepMicrographScreen]

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('noteViz', LabelParam, label="\n")

        form.addParam('visualizeHistogram', IntParam, default=100,
                      label="Visualize Deep Scores Histogram (Bin size)",
                      help="Plot a histogram of the 'goodRegionScore' "
                           "to visualize setting of a good threshold.")
        form.addParam('visualizeCoordinates', FloatParam, default=0.8,
                      label="Visualize good coordinates (threshold from 0 to 1)",
                      help="Visualize the coordinates considered good according"
                           " to the threshold indicated in the box.\n"
                           "If you like the result, save the result"
                           " with the '+Coordinates' button")

    def _getVisualizeDict(self):
        return {'visualizeCoordinates': self._visualizeCoordinates,
                'visualizeHistogram': self._visualizeHistogram}

    def _visualizeCoordinates(self, e=None):
        views = []
        outCoords = self.protocol.getOutput()

        if not outCoords: print(" > Not output found, yet."); return

        coordsViewerFn = self.protocol._getExtraPath('coordsViewer.sqlite')

        mdLabel = emlib.MDL_GOOD_REGION_SCORE

        if not getXmippAttribute(outCoords.getFirstItem(), mdLabel):
            print(" > outputCoordinates do NOT have 'MDL_GOOD_REGION_SCORE'!"); return

        cleanPath(coordsViewerFn)
        newOutput = SetOfCoordinates(filename=coordsViewerFn)
        newOutput.copyInfo(outCoords)
        newOutput.setMicrographs(outCoords.getMicrographs())

        thres = self.visualizeCoordinates.get()
        for coord in outCoords:
            if getXmippAttribute(coord, mdLabel).get() > thres:
                newOutput.append(coord.clone())
        newOutput.write()
        newOutput.close() #SetOfCoordinates does not implement __exit__, required for with

        micSet = newOutput.getMicrographs()  # accessing mics to provide metadata file
        if micSet is None:
            raise Exception('visualize: SetOfCoordinates has no micrographs set.')

        fn = self.protocol._getExtraPath("allMics.xmd")
        xmipp3.convert.writeSetOfMicrographs(micSet, fn)
        tmpDir = self.protocol._getExtraPath('manualThresholding_%03d'%int(thres*100))
        cleanPath(tmpDir)
        makePath(tmpDir)
        xmipp3.convert.writeSetOfCoordinates(tmpDir, newOutput)

        views.append(CoordinatesObjectView(self._project, fn, tmpDir,
                                           self.protocol, inTmpFolder=True))

        return views

    def _visualizeHistogram(self, e=None):
        views = []
        numberOfBins = self.visualizeHistogram.get()

        outCoords = self.protocol.getOutput()
        if outCoords:
            mdLabel = emlib.MDL_GOOD_REGION_SCORE
            if getXmippAttribute(outCoords.getFirstItem(), mdLabel):
                plotter = EmPlotter()
                plotter.createSubPlot("Deep micrograph score",
                                      "Deep micrograph score",
                                      "Number of Coordinates")
                cScores = [getXmippAttribute(coord, mdLabel).get()
                           for coord in outCoords]
                plotter.plotHist(cScores, nbins=numberOfBins)
                views.append(plotter)
            else:
                print(" > 'outputCoordinates' don't have 'xmipp_zScoreDeepLearning2' label.")
        else:
            print(" > Output not ready yet.")

        return views
