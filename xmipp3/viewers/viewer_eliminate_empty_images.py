# **************************************************************************
# *
# * Authors:     Roberto Marabini (roberto@cnb.csic.es)
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

from pyworkflow.em.viewers import EmPlotter, ObjectView, MicrographsView, DataViewer, ClassesView
from pyworkflow.em.viewers.showj import MODE, MODE_MD, ORDER, VISIBLE, RENDER, ZOOM, SORT_BY, MODE_TABLE
from pyworkflow.protocol.params import IntParam, LabelParam
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer

from xmipp3.protocols.protocol_eliminate_empty_images import \
    XmippProtEliminateEmptyClasses, XmippProtEliminateEmptyParticles


class XmippEliminateEmptyViewer(ProtocolViewer):
    """ This protocol computes the maximum resolution up to which two
     CTF estimations would be ``equivalent'', defining ``equivalent'' as having
      a wave aberration function shift smaller than 90 degrees
    """
    _label = 'viewer Eliminate empty images'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [XmippProtEliminateEmptyParticles,
                XmippProtEliminateEmptyClasses]

    def _defineParams(self, form):
        isClassType = isinstance(self.protocol, XmippProtEliminateEmptyClasses)
        self.imagesType = ('Classes' if isClassType else 'Particles')
        form.addSection(label='Visualization')
        form.addParam('visualizeIms', LabelParam,
                       label="Visualize passed %s" % self.imagesType)
        form.addParam('visualizeImsDiscarded', LabelParam,
                        label="Visualize discarded %s" % self.imagesType)

        if not isClassType:
            form.addParam('justSpace', LabelParam, label="")
            form.addParam('visualizeHistogram', IntParam, default=100,
                          label="Visualize Histogram (Bin size)",
                          help="Histogram of the emptiness score.")

    def _getVisualizeDict(self):
        return {
                 'visualizeIms': self._visualizeIms,
                 'visualizeImsDiscarded': self._visualizeImsDiscarded,
                 'visualizeHistogram': self._visualizeHistogram
                }

    def _visualizeImages(self, outputName):
        views = []
        prot = self.protocol
        if hasattr(prot, outputName):
            ouputFn = prot.outputParticles.getFileName()
            ouputId = prot.outputParticles.strId()
            labels = ('id enabled _index _filename _xmipp_scoreEmptiness '
                      '_xmipp_scoreByVariance _xmipp_zScore '
                      '_xmipp_cumulativeSSNR _sampling '
                      '_ctfModel._defocusU _ctfModel._defocusV '
                      '_ctfModel._defocusAngle _transform._matrix')
            if 'Classes' in outputName:
                views.append(ClassesView(self._project, ouputId, ouputFn))
            elif 'Particles' in outputName:
                views.append(ObjectView(self._project, ouputId, ouputFn,
                                        viewParams={ORDER: labels,
                                        VISIBLE: labels,
                                        SORT_BY: '_xmipp_scoreEmptiness asc',
                                        MODE: MODE_MD,
                                        RENDER: '_filename'}))
            else:
                print("Some error occurred, %s not implemented" % outputName)
        else:
            appendStr = ', yet.' if prot.isActive() else '.'
            self.infoMessage('%s does not have %s%s'
                             % (prot.getObjLabel(), outputName,
                                appendStr),
                             title='Info message').show()
        return views

    def _visualizeIms(self, e=None):
        return self._visualizeImages(outputName="output%s" % self.imagesType)

    def _visualizeImsDiscarded(self, e=None):
        return self._visualizeImages(outputName="eliminated%s" % self.imagesType)

    def _visualizeHistogram(self, e=None):
        views = []
        numberOfBins = self.visualizeHistogram.get()
        goodScores = []
        badScores = []
        if hasattr(self.protocol, "outputParticles"):
            goodScores += [part._xmipp_scoreEmptiness.get() for part in
                           self.protocol.outputParticles]

        if hasattr(self.protocol, "eliminatedParticles"):
            badScores += [part._xmipp_scoreEmptiness.get() for part in
                          self.protocol.eliminatedParticles]

        if goodScores or badScores:
            plotter = EmPlotter()
            plotter.createSubPlot("Emptiness Score", "Emptiness Score (a.u.)",
                                  "# of Particles")

            if goodScores:
                w1 = (max(goodScores) - min(goodScores)) / numberOfBins
                plotter.plotHist(goodScores, nbins=numberOfBins, color='green')

            if badScores:
                numberOfBins = (numberOfBins if not goodScores else
                                int((max(badScores) - min(badScores)) / w1))
                plotter.plotHist(badScores, nbins=numberOfBins, color='red')

            if goodScores and badScores:
                plotter.legend(labels=["Passed particles",
                                       "Discarded particles"])

            views.append(plotter)

        return views