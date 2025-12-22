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
from xmippLib import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
        form.addParam('visualizeMaskAndMic', LabelParam,
                    label="Visualize micrographs with the mask overlay",
                    help="Visualize micrographs with the DeepMicrographCleaner mask "
                         "overlaid using a color transparency and the coords stamp.\n"
                         "The transparency indicates the Deep score factor applied."
                         "Will be shown the first 50 miccrographs"
        )
    def _getVisualizeDict(self):
        return {'visualizeCoordinates': self._visualizeCoordinates,
                'visualizeHistogram': self._visualizeHistogram,
                'visualizeMaskAndMic': self._visualizeMaskAndMic}

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
                cScores = [getXmippAttribute(coord, mdLabel).get() for coord in outCoords]
                plotter.plotHist(cScores, nbins=numberOfBins)
                views.append(plotter)
            else:
                print(" > 'outputCoordinates' don't have 'xmipp_zScoreDeepLearning2' label.")
        else:
            print(" > Output not ready yet.")

        return views

    def _visualizeMaskAndMic(self, e=None):
        views = []

        thumb_dir = self.protocol._getExtraPath('thumbnails')
        thumbs = sorted([
            os.path.join(thumb_dir, f)
            for f in os.listdir(thumb_dir)
            if f.lower().endswith('.png')
        ])

        if not thumbs:
            return None

        n = len(thumbs)
        per_page = 15
        cols = 5
        rows = 3

        for page_start in range(0, n, per_page):
            page_thumbs = thumbs[page_start:page_start + per_page]
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)

            try:
                fig.canvas.manager.set_window_title(f'Thumbnails {page_start + 1}-{page_start + len(page_thumbs)}')
            except AttributeError:
                pass

            for ax in axes.flat:
                ax.axis('off')

            for i, fn in enumerate(page_thumbs):
                r, c = divmod(i, cols)
                img = mpimg.imread(fn)
                axes[r, c].imshow(img)
                axes[r, c].axis('off')

            for j in range(len(page_thumbs), rows * cols):
                r, c = divmod(j, cols)
                axes[r, c].imshow(np.zeros((10, 10, 3), dtype=np.uint8))
                axes[r, c].axis('off')


            fig.tight_layout(pad=0.2)
            views.append(fig)

        return views