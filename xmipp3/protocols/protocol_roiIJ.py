# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
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
import os

from pyworkflow.em import *
from pyworkflow.em.viewers.showj import *
from pyworkflow.utils import importFromPlugin

Mesh = importFromPlugin("tomo.objects", "Mesh")
SetOfMeshes = importFromPlugin("tomo.objects", "SetOfMeshes")
TomogramsTreeProvider = importFromPlugin("tomo.viewers.views_tkinter_tree", "TomogramsTreeProvider")
TomogramsDialog = importFromPlugin("tomo.viewers.views_tkinter_tree", "TomogramsDialog")
ProtTomoBase = importFromPlugin("tomo.protocols.protocol_base", "ProtTomoBase")


class XmippProtRoiIJ(ProtAnalysis2D, ProtTomoBase):
    """ Tomogram ROI selection in IJ """
    _label = 'imagej roi'

    def __init__(self, **kwargs):
        ProtAnalysis2D.__init__(self, **kwargs)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('inputTomos', PointerParam, label="Input Tomograms",
                      pointerClass='SetOfTomograms',
                      help='Select tomograms.')

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        # Launch Boxing GUI
        self._insertFunctionStep('launchIJGUIStep', interactive=True)

    def _createOutput(self):
        outSet = self._createSetOfMeshes()
        for file in os.listdir(self._getExtraPath()):
            if file.endswith(".txt"):
                mesh_roi = Mesh(self._getExtraPath(file))
                for tomo in self.inputTomos.get().iterItems():
                    if file[:-5] in tomo.getFileName():
                        mesh_roi.setVolume(tomo)
                outSet.append(mesh_roi)
        outSet.setVolumes(self.inputTomos.get())
        self._defineOutputs(outputMeshes=outSet)
        self._defineSourceRelation(self.inputTomos.get(), outSet)

    # --------------------------- STEPS functions -----------------------------
    def launchIJGUIStep(self):

        tomoList = [tomo.clone() for tomo in self.inputTomos.get().iterItems()]

        tomoProvider = TomogramsTreeProvider(tomoList, self._getExtraPath())

        path = self._getExtraPath()
        self.dlg = TomogramsDialog(None, False, provider=tomoProvider, path=path)

        self._createOutput()


    def _summary(self):
        summary = []
        if not os.listdir(self._getExtraPath()):
            summary.append("Output ROIs not ready yet.")
        else:
            count = 0
            for file in os.listdir(self._getExtraPath()):
                if file.endswith(".txt"):
                    count += 1
            summary.append("Rois defined for %d/%d files have been saved in Scipion (%s)." % (
                count, self.inputTomos.get().getSize(), self._getExtraPath()))
        return summary

    def _methods(self):
        tomos = self.inputTomos.get()
        return [
            "ROI selection and extraction using ImageJ",
            "A total of %d tomograms of dimensions %s were used"
            % (tomos.getSize(), tomos.getDimensions()),
        ]
