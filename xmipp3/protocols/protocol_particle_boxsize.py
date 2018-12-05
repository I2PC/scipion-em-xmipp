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

import csv
import numpy as np

from pyworkflow import VERSION_1_1
from pyworkflow.protocol.params import (PointerParam, StringParam, 
                                        BooleanParam, FloatParam,
                                        LEVEL_ADVANCED)
from pyworkflow.em import ProtMicrographs
from pyworkflow.object import Integer
from pyworkflow.em import ImageHandler
from pyworkflow.utils import getExt
from pyworkflow.em.data import EMObject
import pyworkflow.em.metadata as md
from xmipp3.convert import writeSetOfMicrographs
import xmipp3

from os.path import expanduser # TODO: can be removed when moving the weights/scaler to scipion folders


class XmippProtParticleBoxsize(ProtMicrographs):
    """    
    Given a map the protocol assigns local resolutions to each voxel of the map.
    """
    _label = 'particle_boxsize'
    _lastUpdateVersion = VERSION_1_1
    
    def __init__(self, **args):
        ProtMicrographs.__init__(self, **args)
        self.particle_boxsize = None
    
    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')


        form.addParam('micrographs', PointerParam, pointerClass='SetOfMicrographs',
                      label="Input Micrographs", important=True,
                      help='Select a set of micrographs for determining the '
                      'particle boxsize.')

    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
            # Convert input into xmipp Metadata format
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('boxsizeStep')
        self._insertFunctionStep('createOutputStep')

    def convertInputStep(self):
        writeSetOfMicrographs(self.micrographs.get(),
                              self._getExtraPath('input_micrographs.xmd'))

    def boxsizeStep(self):
        # TODO: hardcoded values: this should be changed!! 
        FN_PARTICLE_BOXSIZE = self._getExtraPath('particle_boxsize.xmd')
        img_size = 300  # This should match the img_size used for training weights? Add as metada?
        weights = xmipp3.Plugin.getHDF5model('boxsize/weights.hdf5')  #    expanduser('/home/vilas/Downloads/boxsize/weights.hdf5')
        feature_scaler = xmipp3.Plugin.getHDF5model('boxsize/feature_scaler.pkl')  #expanduser('/home/vilas/Downloads/boxsize/feature_scaler.pkl')
        base_params = ' --img_size %d' % img_size
        base_params += ' --weights %s' % weights
        base_params += ' --feature_scaler %s' % feature_scaler
        base_params += ' --output %s' % FN_PARTICLE_BOXSIZE
        
        
        filenames = [mic.getFileName() + '\n' for mic in self.micrographs.get()]
        # TODO: output name is hardcoded
        mic_names_path = self._getTmpPath('mic_names.csv')
        with open(mic_names_path, 'wb') as csvfile:
            csvfile.writelines(filenames)
        params = base_params + ' --micrographs %s' % mic_names_path
        self.runJob('xmipp_particle_boxsize', params)
        
        with open(FN_PARTICLE_BOXSIZE, 'r') as fp:
            self.particle_boxsize = int(fp.read().rstrip('\n'))
        
        print(self.particle_boxsize)

       
    def createOutputStep(self):
        output = EMObject()
        output.boxsize = Integer(self.particle_boxsize)
        
        self._defineOutputs(boxsize=output)

                            

    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        if hasattr(self, 'resolution_Volume'):
            messages.append(
                'Information about the method/article in ' + MONORES_METHOD_URL)
        return messages
    
    def _summary(self):
        summary = []
        return summary

    def _citations(self):
        return ['']

