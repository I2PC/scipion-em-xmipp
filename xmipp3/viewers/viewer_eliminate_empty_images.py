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

from pyworkflow.protocol.params import IntParam, LabelParam
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
from pwem.viewers import EmPlotter, ObjectView, ClassesView
from pwem.viewers.showj import MODE, MODE_MD, ORDER, VISIBLE, RENDER, SORT_BY

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
        self.imagesType = ('Averages' if isClassType else 'Particles')
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
            ouputFn = getattr(prot, outputName).getFileName()
            ouputId = getattr(prot, outputName).strId()
            labels = ('id enabled _index _filename _xmipp_scoreEmptiness '
                      '_xmipp_scoreByVariance _xmipp_zScore '
                      '_xmipp_cumulativeSSNR _sampling '
                      '_ctfModel._defocusU _ctfModel._defocusV '
                      '_ctfModel._defocusAngle _transform._matrix')
            if 'Classes' in outputName:
                views.append(ClassesView(self._project, ouputId, ouputFn))
            else:
                views.append(ObjectView(self._project, ouputId, ouputFn,
                                        viewParams={ORDER: labels,
                                        VISIBLE: labels,
                                        SORT_BY: '_xmipp_scoreEmptiness desc',
                                        MODE: MODE_MD,
                                        RENDER: '_filename'}))
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

        plotter = EmPlotter()
        plotter.createSubPlot("Emptiness Score", "Emptiness Score (a.u.)",
                              "# of Particles")

        values = [goodScores, badScores]
        labels = ["Passed particles", "Discarded particles"]
        colors = ['green', 'red']

        plotMultiHistogram(values, colors, labels, numberOfBins, plotter, views)

        return views


def plotMultiHistogram(valuesList, colors=None, legend=None, numOfBins=100,
                       plotter=None, views=None, includeEmpties=False):
    """ Values list must be a n-list of list,
        where n is the number of the subhistograms to plot.
        Multiple histograms will be plot in the same chart
        If no views is passed, a new list-views will be returned with the hist.
        If no plotter is passed, a new generic one will be created.
    """

    if not all([isinstance(x, list) for x in valuesList]):
        print("Not all items in values list are lists. Returning...")
        return

    if colors is None:
        from matplotlib import colors
        from random import shuffle
        colors = colors.cnames.keys()
        shuffle(colors)


    if any([len(x) for x in valuesList]):
        if plotter is None:
            plotter = EmPlotter()
            plotter.createSubPlot("Histogram", "Score", "# of Items")

        w1 = None
        finalLegend = []
        for idx, values in enumerate(valuesList):
            if values or includeEmpties:
                if w1 is None:
                    w1 = (max(values) - min(values)) / numOfBins
                else:
                    numOfBins = int((max(values) - min(values)) / w1)

                plotter.plotHist(values, nbins=numOfBins, color=colors[idx])
                if legend:
                    finalLegend.append(legend[idx])

        if finalLegend:
            plotter.legend(labels=finalLegend)

        if views is None:
            views = [plotter]
        else:
            views.append(plotter)

        return views