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

import pyworkflow
import pyworkflow.object as pwobj
from pyworkflow.em import *
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.em.data import Particle

import xmippLib


class XmippProtProjectZ(ProtAnalysis3D):
    """
    Project a set of volumes or a set of subtomograms to obtain their
    top view.
    """
    _label = 'Z projection'
    _version = pyworkflow.VERSION_1_1
    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('input', PointerParam, pointerClass="SetOfSubTomograms, SetOfVolumes",
                      label='Input Volumes/Subtomograms')

    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('projectZStep')
        self._insertFunctionStep('createOutputStep')
    
    #--------------------------- STEPS functions -------------------------------
    def projectZStep(self):
        idx = 1
        x, y, _ = self.input.get().getDim()
        fnProj = self._getExtraPath("projections.mrcs")
        xmippLib.createEmptyFile(fnProj, x, y, 1, self.input.get().getSize())

        for item in self.input.get().iterItems():
            self.runJob("xmipp_phantom_project",
                        "-i %d@%s -o %s --angles 0 0 0" %
                        (item.getIndex(), item.getFileName(), "%d@%s" % (idx,fnProj)))
            idx += 1

    def createOutputStep(self):
        imgSetOut = self._createSetOfAverages()
        imgSetOut.setSamplingRate(self.input.get().getSamplingRate())
        imgSetOut.setAlignmentProj()

        fnProj = self._getExtraPath("projections.mrcs")
        for idv in range(self.input.get().getSize()):
            p = Particle()
            p.setLocation(idv+1,fnProj)
            imgSetOut.append(p)

        self._defineOutputs(outputReprojections=imgSetOut)
        self._defineSourceRelation(self.input, imgSetOut)
