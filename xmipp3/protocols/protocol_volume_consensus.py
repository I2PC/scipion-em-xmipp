# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:  Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
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

from os.path import exists
from pyworkflow.protocol.params import MultiPointerParam
from pwem.objects import Volume, Transform
from pwem.protocols import ProtInitialVolume

CITE = 'Fernandez-Gimenez2021'

class XmippProtVolConsensus(ProtInitialVolume):
    """ This protocol performs a fusion of all the input volumes, which should be preprocessed with protocol 'volume
    substraction' saving volume 2, in order to be as similar as possible before the fusion. The output of
    this protocol is the consensus volume and another volume which indicates the maximun difference between input
    volumes in each voxel."""

    _label = 'volume consensus'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('vols', MultiPointerParam, pointerClass='Volume', label="Volumes",
                      help='Select the volumes for the consensus.')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('fusionStep')
        self._insertFunctionStep('createOutputStep')
        self._insertFunctionStep("createChimeraScript")

    # --------------------------- STEPS functions ---------------------------------------------------
    def fusionStep(self):
        inputVols = self._getExtraPath("input_volumes.txt")
        fhInputVols = open(inputVols, 'w')
        for i, vol in enumerate(self.vols):
            fileName = vol.get().getFileName()
            if fileName.endswith(':mrc'):
                fileName = fileName[:-4]
            fhInputVols.write(fileName + '\n')
        fhInputVols.close()
        outVolFn = self._getExtraPath("consensus_volume.mrc")
        args = " -i %s -o %s" % (inputVols, outVolFn)
        self.runJob("xmipp_volume_consensus", args)

    def createOutputStep(self):
        outVol = Volume()
        outVol.setSamplingRate(self.vols[0].get().getSamplingRate())
        outVol.setFileName(self._getExtraPath("consensus_volume.mrc"))
        if not exists(self._getExtraPath("consensus_volume.mrc")):
            raise NoOutputGenerated("Consensus volume NOT generated, please check input volumes to ensure they have "
                                    "equal box size, voxel size and origin.")
        else:
            outVol2 = Volume()
            outVol2.setSamplingRate(self.vols[0].get().getSamplingRate())
            outVol2.setFileName(self._getExtraPath("consensus_volume_diff.mrc"))
            self._defineOutputs(outputVolume=outVol)
            self._defineOutputs(outputVolumeDiff=outVol2)

    def createChimeraScript(self):
        fnRoot = "extra/"
        scriptFile = self._getPath('result') + '_fusion_chimera.cxc'
        fhCmd = open(scriptFile, 'w')
        fhCmd.write("open %s\n" % (fnRoot+"consensus_volume.mrc"))
        fhCmd.write("open %s\n" % (fnRoot+"consensus_volume_diff.mrc"))
        fhCmd.write("vol #2 hide\n")
        fhCmd.write("color sample #1 map #2 palette rainbow\n")
        fhCmd.close()

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

    def _validate(self):
        errors = []
        voxel_size = []
        for i, vol in enumerate(self.vols):
            voxel_size.append(round(vol.get().getSamplingRate(), 2))
        result = all(element == voxel_size[0] for element in voxel_size)
        if not result:
            errors.append('Pixel size should be the same for all input volumes.')
        return errors

    def _citations(self):
        return ['Fernandez-Gimenez2021']


class NoOutputGenerated(Exception):
    """No output generation error"""
    pass
