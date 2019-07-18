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

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import PointerParam
from pyworkflow.em import ProtMicrographs
from pyworkflow.object import Integer

from xmipp3.convert import writeSetOfMicrographs
import xmipp3
from xmipp3.utils import validateDLtoolkit


class XmippProtParticleBoxsize(ProtMicrographs):
    """ Given a set of micrographs, the protocol estimate the particle box size.
    """
    _label = 'particle boxsize'
    _lastUpdateVersion = VERSION_2_0
    
    def __init__(self, **args):
        ProtMicrographs.__init__(self, **args)
        self.particleBoxsize = None
    
    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputMicrographs', PointerParam, important=True,
                      label="Input Micrographs", pointerClass='SetOfMicrographs',
                      help='Select a set of micrographs for determining the '
                           'particle boxsize.')

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        # Convert input into xmipp Metadata format
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('boxsizeStep')
        self._insertFunctionStep('createOutputStep')

    def convertInputStep(self):
        writeSetOfMicrographs(self.inputMicrographs.get(),
                              self._getExtraPath('input_micrographs.xmd'))

    def boxsizeStep(self):
        particleBoxSizeFn = self._getExtraPath('particle_boxsize.xmd')
        imgSize = 300  # This should match the img_size used for training weights? Add as metada?
        weights = xmipp3.Plugin.getModel('boxsize', 'weights.hdf5')
        featureScaler = xmipp3.Plugin.getModel('boxsize', 'feature_scaler.pkl')

        params  = ' --img_size %d' % imgSize
        params += ' --weights %s' % weights
        params += ' --feature_scaler %s' % featureScaler
        params += ' --output %s' % particleBoxSizeFn

        fileNames = [mic.getFileName() + '\n' for mic in self.inputMicrographs.get()]
        # TODO: output name is hardcoded
        micNamesPath = self._getTmpPath('mic_names.csv')
        with open(micNamesPath, 'wb') as csvFile:
            csvFile.writelines(fileNames)
        params += ' --micrographs %s' % micNamesPath
        self.runJob('xmipp_particle_boxsize', params)
        
        with open(particleBoxSizeFn, 'r') as fp:
            self.particleBoxsize = int(fp.read().rstrip('\n'))

        print("\n > Estimated box size: %d \n" % self.particleBoxsize)

    def createOutputStep(self):
        """ The output is just an Integer. Other protocols can use it in those
            IntParam if it has set allowsPointer=True
        """
        micSet = self.inputMicrographs.get()
        boxSize = Integer(self.particleBoxsize)
        self._defineOutputs(boxsize=boxSize)
        self._defineSourceRelation(micSet, boxSize)

    # --------------------------- INFO functions ------------------------------
    def _methods(self):
        messages = []
        if hasattr(self, 'boxsize'):
            messages.append('Estimated box size: %s pixels' % self.boxsize)
        return messages
    
    def _summary(self):
        messages = []
        if hasattr(self, 'boxsize'):
            messages.append('Estimated box size: %s pixels' % self.boxsize)
            messages.append("Open 'Analyze results' to see it in context. "
                            "(The displayed coordinate is just to show the size)")
        return messages

    def _citations(self):
        return ['']

    def _validate(self):
        return validateDLtoolkit(model=[('boxsize', 'weights.hdf5'),
                                        ('boxsize', 'feature_scaler.pkl')])
