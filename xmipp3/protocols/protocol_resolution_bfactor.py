# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Jose Luis Vilas (jlvilas@cnb.csic.es)
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

import numpy as np

from pwem.emlib.image import ImageHandler
from pwem.objects import FSC
from pwem.protocols import ProtAnalysis3D
from pyworkflow import VERSION_2_0
from pyworkflow.protocol.constants import STEPS_PARALLEL
import pyworkflow.protocol.params as params
import pwem.emlib.metadata as md

from xmipp3.convert import locationToXmipp, writeSetOfParticles

from pyworkflow.protocol.params import (PointerParam, BooleanParam, FloatParam,
                                        LEVEL_ADVANCED)
from pyworkflow.object import Float
from pyworkflow.utils import getExt
from pwem.objects import Volume, SetOfParticles


FN_METADATA_BFACTOR_RESOLUTION = 'bfactor_resolution.xmd'

class XmippProtbfactorResolution(ProtAnalysis3D):
    """    
    Given a local resolution map the matching of the local resolution with the bfactor of each atom
    taken from an atomic model is estimated. In particular, the matching of the bfactor and local resolution
    per residue.
    """
    _label = 'resolution bfactor'
    _lastUpdateVersion = VERSION_2_0

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('pdbfile', PointerParam, pointerClass='AtomStruct',
                      label="Atomic model", important=True,
                      help='Select an atomic model. The atom positions will be taken'
                           ' to estimate the local resolution around them and then, the '
                           ' local resolution associated to each residue.')

        form.addParam('normalizeResolution', BooleanParam, default=True,
                      label="Normalize Resolution",
                      help='The local resolution normalized map is defined as'
                           '(LR - FSC)/FSC, where LR is the local resolution of a'
                           'given voxel, and FSC is the FSC resolution in A. This '
                           'map provides information about if the local resolution'
                           'greater or lesser than the FSC. The local resolution '
                           'normalized map is used to carry out the matching with the local'
                           'bfactor per residue.')

        form.addParam('localResolutionMap', PointerParam, pointerClass='Volume',
                      label="Local Resolution Map", important=True,
                      condition='normalizeResolution',
                      help='Select a local resolution map. Alternatively, the input.'
                           ' can be a normalized local resolution map, in this case'
                           'set the Normalize resolution to No')

        form.addParam('normalizedMap', PointerParam, pointerClass='Volume',
                      label="Normalized Local Resolution Map", important=True,
                      condition='not normalizeResolution',
                      help='Select a normalized local resolution map. The local'
                           ' resolution normalized map is defined as '
                           '(LR - FSC)/FSC, where LR is the local resolution of a'
                           'given voxel, and FSC is the FSC resolution in A')

        form.addParam('fscResolution', FloatParam,
                      condition = 'normalizeResolution',
                      label="FSC resolution (A)",
                      help='The global resolution of the map in A')

        form.addParam('medianEstimation', BooleanParam, default=True,
                      label="Use median",
                      help='The local resolution per residue can be estimated using'
                           'the mean (by default) or the median')

        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):

        # 1 Check the input and convert to mrc if it is the case
        self._insertFunctionStep('convertInputStep')

        # 2 Carry out the mathing betweent he local resolution per residue and bfactor
        self._insertFunctionStep('matchingBfactorLocalResolution')

        # Create the scipion output
        # self._insertFunctionStep('createOutputStep')

    def mrc_convert(self, fileName, outputFileName):
        """Check if the extension is .mrc, if not then uses xmipp to convert it
        """
        ext = getExt(fileName)
        if ((ext != '.mrc') or (ext != '.map')):
            params = ' -i %s' % fileName
            params += ' -o %s' % outputFileName
            self.runJob('xmipp_image_convert', params)
            return outputFileName
        else:
            return fileName

    def convertInputStep(self):
        """ Read the input volume and check the file extension to convert to mrc is it is the case.
        """
        self.vol = self.mrc_convert(self.localResolutionMap.get().getFileName(),
                                    self._getTmpPath('localResolutionMap.mrc'))

    def matchingBfactorLocalResolution(self):
        """ The local resolution map and the pdb are taken and analyzed to match the
        local resolution and bfactor per residue. The output will be a pdb file with the
        bfactor column substituted by the normalized local resolution. This is the (local
        resolution - fscResolution)/fscResolution.
        """

        params = ' --atmodel %s' % self.pdbfile.get().getFileName()
        params += ' --vol %s' % self.vol
        if self.normalizeResolution.get() is True:
            params += ' --fscResolution %s' % self.fscResolution.get()
        params += ' --sampling %f' % self.localResolutionMap.get().getSamplingRate()
        if self.medianEstimation.get() is True:
            params += ' --hasMedian '
        params += ' -o %s' % self._getExtraPath()

        self.runJob('xmipp_resolution_pdb_bfactor', params)

    """ 
    def createOutputStep(self):
        volume=Volume()
        volume.setFileName(self._getExtraPath("3dFSC.mrc"))

        volume.setSamplingRate(self.half1.get().getSamplingRate())
        self._defineOutputs(fsc3D=volume)
        self._defineSourceRelation(self.half1, volume)
    """

    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        if hasattr(self, 'Volume'):
            messages.append(
                'Information about the method/article in ')
        return messages

    def _summary(self):
        summary = []
        summary.append(" ")
        return summary

    def _citations(self):
        return ['Vilas2020']
