# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
# *              Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
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

from pyworkflow.em.data import Particle
from pyworkflow.em.protocol import ProtAnalysis3D
from pyworkflow.protocol.params import PointerParam, EnumParam, IntParam
from pyworkflow.utils import importFromPlugin
SetOfTomograms = importFromPlugin("tomo.objects", "SetOfTomograms")
import xmippLib

class XmippProtProjectZ(ProtAnalysis3D):
    """
    Project a set of volumes or subtomograms to obtain their X, Y or Z projection of the desired range of slices.
    """
    _label = 'Projection'

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('input', PointerParam, pointerClass="SetOfSubTomograms, SetOfVolumes",
                      label='Input Volumes')
        form.addParam('dirParam', EnumParam, choices=['X', 'Y', 'Z'], default=2, display=EnumParam.DISPLAY_HLIST,
                      label='Projection direction')
        form.addParam('rangeParam', EnumParam, choices=['All', 'Range'], default=0, display=EnumParam.DISPLAY_HLIST,
                      label='Range of slices', help='Range of slices used to compute the projection, where 0 is the '
                                                    'central slice.')
        form.addParam('cropParam', IntParam, default=10, label='Voxels', condition="rangeParam == 1",
                      help='Crop this amount of voxels in the selected direction. Half of the pixels will be cropped '
                           'from one side and the other half from the other. If X direction is selected, the crop will '
                           'be performed in X, Y and Z.')

    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('projectZStep')
        self._insertFunctionStep('createOutputStep')
    
    # --------------------------- STEPS functions -------------------------------
    def projectZStep(self):
        x, y, z = self.input.get().getDim()
        n = self.input.get().getSize()
        fnInput = self._getExtraPath("input.stk")
        fnWin = self._getExtraPath("window.stk")
        fnProj = self._getExtraPath("projection.stk")
        xmippLib.createEmptyFile(fnInput, x, y, z, n)

        if self.rangeParam.get() == 0: cropParam = 0
        else: cropParam = self.cropParam.get()

        if self.dirParam.get() == 2:  # Z
            angles = '0 0 0'
            crop = '0 0 %d' % cropParam
            xmippLib.createEmptyFile(fnWin, x, y, z-cropParam, n)
            xmippLib.createEmptyFile(fnProj, x, y, 1, n)
        elif self.dirParam.get() == 1:  # Y
            angles = '90 90 0'
            crop = '0 %d 0' % cropParam
            xmippLib.createEmptyFile(fnWin, x, y-cropParam, z, n)
            xmippLib.createEmptyFile(fnProj, x, y-cropParam, 1, n)
        else:  # X
            angles = '0 90 0'
            crop = '%d 0 0' % cropParam
            xmippLib.createEmptyFile(fnWin, x-cropParam, y-cropParam, z-cropParam, n)
            xmippLib.createEmptyFile(fnProj, x-cropParam, y-cropParam, 1, n)

        for item in self.input.get().iterItems():
            idx = item.getIndex()
            if type(self.input.get()) == SetOfTomograms: idx = idx + 1
            self.runJob("xmipp_image_convert", "-i %d@%s -o %d@%s -a" % (idx, item.getFileName(), idx, fnInput))
            self.runJob("xmipp_transform_window", "-i %d@%s -o %d@%s --crop %s" % (idx, fnInput, idx, fnWin, crop))
            self.runJob("xmipp_phantom_project", "-i %d@%s -o %d@%s --angles %s" % (idx, fnWin, idx, fnProj, angles))

    def createOutputStep(self):
        imgSetOut = self._createSetOfAverages()
        imgSetOut.setSamplingRate(self.input.get().getSamplingRate())
        imgSetOut.setAlignmentProj()
        fnProj = self._getExtraPath("projections.mrcs")
        for idv in range(self.input.get().getSize()):
            p = Particle()
            p.setLocation(idv+1, fnProj)
            imgSetOut.append(p)

        imgSetOut.setObjComment(self.getSummary(imgSetOut))
        self._defineOutputs(outputReprojections=imgSetOut)
        self._defineSourceRelation(self.input, imgSetOut)

# --------------------------- INFO functions ------------------------------
    def _methods(self):
        vols = self.input.get()
        return ["Projection of %d volumes with dimensions %s obtained with xmipp_phantom_project"
                % (vols.getSize(), vols.getDimensions())]

    def _summary(self):
        summary = []
        if not self.isFinished():
            summary.append("Output views not ready yet.")

        if self.getOutputsSize() >= 1:
            for key, output in self.iterOutputAttributes():
                summary.append("*%s:* \n %s " % (key, output.getObjComment()))
        return summary

    def getSummary(self, imgSetOut):
        summary = []
        summary.append("\n   -Number of projections generated: %s" % imgSetOut.getSize())
        return "\n".join(summary)
