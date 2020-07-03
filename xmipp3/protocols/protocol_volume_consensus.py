# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:  Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
# *           Ruben Sanchez (rsanchez@cnb.csic.es)
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

from pyworkflow.protocol.params import MultiPointerParam
from pwem.convert import headers
from pwem.objects import Volume, Transform
from pwem.protocols import ProtInitialVolume
from pyworkflow.utils import removeExt


class XmippProtVolConsensus(ProtInitialVolume):
    """ This protocol performs a fusion of all the input volumes, which should be preprocessed with protocol
    "volume substraction" saving volume 2, in order to be as similar as possible before the fusion. """

    _label = 'volume consensus'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):

        form.addSection(label='Input')
        form.addParam('vols', MultiPointerParam, pointerClass='Volume', label="Volumes",
                      help='Select the volumes for the consensus.')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('fusionStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertInputStep(self):
        pass


    def fusionStep(self):
        inputVols = self._getExtraPath("input_volumes.txt")
        fhInputVols = open(inputVols, 'w')
        for i, vol in enumerate(self.vols):
            fileName = vol.get().getFileName()
            if fileName.endswith(':mrc'):
                fileName = fileName[:-4]
            # if not fileName.endswith('.mrc'):
            #     volmrc = self._getExtraPath('vol_%d.mrc' % i)
            #     args = ' -i %s -o %s' % (fileName, volmrc)
            #     program = "xmipp_image_convert"
            #     self.runJob(program, args)
            #     fileName = volmrc
            fhInputVols.write(fileName + '\n')
        fhInputVols.close()
        outVolFn = self._getExtraPath("consensus_volume.mrc")
        args = " -i %s -o %s" % (inputVols, outVolFn)
        self.runJob("xmipp_volume_consensus", args)

    def createOutputStep(self):
        outVol = Volume()
        outVol.setSamplingRate(self.vols[0].get().getSamplingRate())
        outVol.setFileName(self._getExtraPath("consensus_volume.mrc"))
        # origin = Transform()
        # ccp4header = headers.Ccp4Header(self.vols[0].get().getFileName(), readHeader=True)
        # shifts = ccp4header.getOrigin()
        # origin.setShiftsTuple(shifts)
        # outVol.setOrigin(origin)
        self._defineOutputs(outputVolume=outVol)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputVolume'):
            summary.append("Output volume not ready yet.")
        else:
            for i, vol in enumerate(self.vols):
                summary.append("Volume %d: %s" % (i+1, vol.get().getFileName()))
        return summary

    def _methods(self):
        methods = []
        if not hasattr(self, 'outputVolume'):
            methods.append("Output volume not ready yet.")
        else:
            methods.append("We compute a consensus volume from %d input volumes at %f A/px" %
                           (self.vols.getSize(), self.vols[0].get().getSamplingRate()))
        return methods
