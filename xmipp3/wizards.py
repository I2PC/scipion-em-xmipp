# **************************************************************************
# *
# * Authors:     Jose Gutierrez (jose.gutierrez@cnb.csic.es) [1]
# * Authors:     J.M. De la Rosa Trevin (delarosatrevin@scilifelab.se) [2]
# *
# * [1] Unidad de Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# * [2] SciLifeLab, Stockholm University
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

from pyworkflow.em.constants import *
from pyworkflow.em.wizard import *
from pyworkflow.em.protocol import ProtImportCoordinates

from .constants import *
from .protocols.protocol_cl2d import IMAGES_PER_CLASS

from .protocols import (
    XmippProtCTFMicrographs, XmippProtProjMatch, XmippProtPreprocessParticles,
    XmippProtPreprocessMicrographs, XmippProtPreprocessVolumes,
    XmippProtExtractParticles, XmippProtExtractParticlesPairs,
    XmippProtFilterParticles, XmippProtFilterVolumes, XmippProtMaskParticles,
    XmippProtMaskVolumes, XmippProtAlignVolume, XmippProtCL2D,
    XmippProtHelicalParameters, XmippProtConsensusPicking, XmippProtMonoRes,
    XmippProtRotSpectra, XmippProtReconstructHighRes, XmippProtExtractUnit,
    XmippProtReconstructHeterogeneous, XmippMetaProtDiscreteHeterogeneityScheduler)


#===============================================================================
# DOWNSAMPLING
#===============================================================================

class XmippDownsampleWizard(DownsampleWizard):
    _targets = [(XmippProtPreprocessMicrographs, ['downFactor'])]

    def _getParameters(self, protocol):

        label, value = self._getInputProtocol(self._targets, protocol)

        protParams = {}
        protParams['input']= protocol.inputMicrographs
        protParams['label']= label
        protParams['value']= value

        return protParams

    def _getProvider(self, protocol):
        _objs = self._getParameters(protocol)['input']
        return DownsampleWizard._getListProvider(self, _objs)

    def show(self, form):
        params = self._getParameters(form.protocol)
        _value = params['value']
        _label = params['label']
        DownsampleWizard.show(self, form, _value, _label, UNIT_PIXEL)

#===============================================================================
# CTFS
#===============================================================================

class XmippCTFWizard(CtfWizard):
    _targets = [(XmippProtCTFMicrographs, ['ctfDownFactor', 'lowRes', 'highRes'])]

    def _getParameters(self, protocol):

        label, value = self._getInputProtocol(self._targets, protocol)

        protParams = {}
        protParams['input']= protocol.inputMicrographs
        protParams['label']= label
        protParams['value']= value
        return protParams

    def _getProvider(self, protocol):
        _objs = self._getParameters(protocol)['input']
        return CtfWizard._getListProvider(self, _objs)

    def getAutodownsampling(self,protocol, coeff=1.5):
        samplingRate = protocol.inputMicrographs.get().getSamplingRate()
        downsampling_factor = coeff / samplingRate
        if downsampling_factor < 1:
            downsampling_factor = 1
        return downsampling_factor

    def show(self, form):
        protocol = form.protocol
        params = self._getParameters(protocol)
        _value = params['value']
        _label = params['label']

        if protocol.AutoDownsampling:
            downSampling = self.getAutodownsampling(protocol)
        else:
            downSampling = _value[0]

#        form.setParamFromVar('inputMicrographs') # update selected input micrographs
        provider = self._getProvider(protocol)

        if provider is not None:
            args = {'unit': UNIT_PIXEL,
                    'downsample': downSampling,
                    'lf': _value[1],
                    'hf': _value[2],
                    'showInAngstroms': True
                    }
            oldDowsample = downSampling
            d = CtfDownsampleDialog(form.root, provider, **args)

            if d.resultYes():
                newDownsample=d.getDownsample()
                form.setVar(_label[0], newDownsample)
                form.setVar(_label[1], d.getLowFreq())
                form.setVar(_label[2], d.getHighFreq())
                if abs(newDownsample-oldDowsample)>1e-4:
                    form.setVar("AutoDownsampling",False)
        else:
            dialog.showWarning("Empty input", "Select elements first", form.root)

    @classmethod
    def getView(self):
        return "wiz_ctf_downsampling"

#===============================================================================
# BOXSIZE
#===============================================================================
class XmippBoxSizeWizard(Wizard):
    _targets = [(XmippProtExtractParticles, ['boxSize'])]

    def show(self, form):
        form.setVar('boxSize', form.protocol.getBoxSize())


#===============================================================================
# CONSENSUS RADIUS
#===============================================================================
class XmippParticleConsensusRadiusWizard(Wizard):
    _targets = [(XmippProtConsensusPicking, ['consensusRadius'])]

    def _getRadius(self, protocol):
        if protocol.inputCoordinates.hasValue():
            boxSize=protocol.inputCoordinates[0].get().getBoxSize()
            radius = int(boxSize*0.1)
            if radius<10:
                radius=10
        else:
            radius = 10
        return radius

    def show(self, form):
        form.setVar('consensusRadius', self._getRadius(form.protocol))

#===============================================================================
# NUMBER OF CLASSES
#===============================================================================
class XmippCL2DNumberOfClassesWizard(Wizard):
    _targets = [(XmippProtCL2D, ['numberOfClasses'])]

    def _getNumberOfClasses(self, protocol):

        numberOfClasses = 64

        if protocol.inputParticles.hasValue():
            numberOfClasses = int(protocol.inputParticles.get().getSize()/IMAGES_PER_CLASS)

        return numberOfClasses


    def show(self, form):
        form.setVar('numberOfClasses', self._getNumberOfClasses(form.protocol))

#===============================================================================
# MASKS
#===============================================================================

class XmippParticleMaskRadiusWizard(ParticleMaskRadiusWizard):
    _targets = [(XmippProtMaskParticles, ['radius']),
                (XmippProtPreprocessParticles, ['backRadius']),
                (XmippProtReconstructHighRes, ['particleRadius']),
                (XmippProtReconstructHeterogeneous, ['particleRadius']),
                (XmippMetaProtDiscreteHeterogeneityScheduler, ['particleRadius'])]

    def _getParameters(self, protocol):

        label, value = self._getInputProtocol(self._targets, protocol)

        protParams = {}
        protParams['input']= protocol.inputParticles
        protParams['label']= label
        protParams['value']= value
        return protParams

    def _getProvider(self, protocol):
        _objs = self._getParameters(protocol)['input']
        return ParticleMaskRadiusWizard._getListProvider(self, _objs)

    def show(self, form):
        params = self._getParameters(form.protocol)
        _value = params['value']
        _label = params['label']
        ParticleMaskRadiusWizard.show(self, form, _value, _label, UNIT_PIXEL)



class XmippParticleMaskRadiiWizard(ParticlesMaskRadiiWizard):
    _targets = [(XmippProtMaskParticles, ['innerRadius', 'outerRadius']),
                (XmippProtRotSpectra, ['spectraInnerRadius', 'spectraOuterRadius'])]

    def _getParameters(self, protocol):

        label, value = self._getInputProtocol(self._targets, protocol)

        protParams = {}
        protParams['input']= protocol.inputParticles
        protParams['label']= label
        protParams['value']= value
        return protParams

    def _getProvider(self, protocol):
        _objs = self._getParameters(protocol)['input']
        return ParticlesMaskRadiiWizard._getListProvider(self, _objs)

    def show(self, form):
        params = self._getParameters(form.protocol)
        _value = params['value']
        _label = params['label']
        ParticlesMaskRadiiWizard.show(self, form, _value, _label, UNIT_PIXEL)


class XmippVolumeMaskRadiusBasicWizard(VolumeMaskRadiusWizard):
    def _getParameters(self, protocol):
        label, value = self._getInputProtocol(self._targets, protocol)
        protParams = {}
        protParams['label']= label
        protParams['value']= value
        return protParams

    def _getProvider(self, protocol):
        _objs = self._getParameters(protocol)['input']
        return VolumeMaskRadiusWizard._getListProvider(self, _objs)

    def show(self, form):
        params = self._getParameters(form.protocol)
        _value = params['value']
        _label = params['label']
        VolumeMaskRadiusWizard.show(self, form, _value, _label, UNIT_PIXEL)


class XmippVolumeMaskRadiusWizard(XmippVolumeMaskRadiusBasicWizard):
    _targets = [(XmippProtMaskVolumes, ['radius']),
                (XmippProtAlignVolume, ['maskRadius']),
                (XmippProtPreprocessVolumes, ['backRadius'])]

    def _getParameters(self, protocol):
        protParams=XmippVolumeMaskRadiusBasicWizard._getParameters(self, protocol)
        protParams['input']= protocol.inputVolumes
        return protParams


class XmippVolumeMaskRadiusWizard2(XmippVolumeMaskRadiusBasicWizard):
    _targets = [(XmippProtMonoRes, ['volumeRadius'])]

    def _getParameters(self, protocol):
        protParams=XmippVolumeMaskRadiusBasicWizard._getParameters(self, protocol)
        protParams['input']= protocol.inputVolumes
        return protParams


class XmippVolumeMaskRadiusWizard3(XmippVolumeMaskRadiusBasicWizard):
    _targets = [(XmippProtMonoRes, ['volumeRadiusHalf'])]

    def _getParameters(self, protocol):
        protParams=XmippVolumeMaskRadiusBasicWizard._getParameters(self, protocol)
        protParams['input']= protocol.inputVolumes
        return protParams


class XmippVolumeOuterRadiusWizard(XmippVolumeMaskRadiusWizard):
    _targets = [(XmippProtHelicalParameters, ['cylinderOuterRadius'])]

    def _getParameters(self, protocol):
        protParams = {}
        protParams['input']= protocol.inputVolume
        protParams['label']= 'cylinderOuterRadius'
        protParams['value']= protocol.cylinderOuterRadius.get()
        return protParams


class XmippVolumeInnerRadiusWizard(XmippVolumeMaskRadiusWizard):
    _targets = [(XmippProtHelicalParameters, ['cylinderInnerRadius'])]

    def _getParameters(self, protocol):
        protParams = {}
        protParams['input']= protocol.inputVolume
        protParams['label']= 'cylinderInnerRadius'
        protParams['value']= protocol.cylinderInnerRadius.get()
        return protParams


class XmippVolumeMaskRadiusProjMWizard(XmippVolumeMaskRadiusWizard):
    _targets = [(XmippProtProjMatch, ['maskRadius'])]

    def _getParameters(self, protocol):

        label, value = self._getInputProtocol(self._targets, protocol)

        protParams = {}
        protParams['input']= protocol.input3DReferences
        protParams['label']= label
        protParams['value']= value
        return protParams


class XmippVolumeRadiiWizard(VolumeMaskRadiiWizard):
    _targets = [(XmippProtMaskVolumes, ['innerRadius', 'outerRadius']),
               (XmippProtExtractUnit, ['innerRadius', 'outerRadius'])
              ]

    def _getParameters(self, protocol):

        label, value = self._getInputProtocol(self._targets, protocol)

        protParams = {}
        protParams['input']= protocol.inputVolumes
        protParams['label']= label
        protParams['value']= value
        return protParams

    def _getProvider(self, protocol):
        _objs = self._getParameters(protocol)['input']
        return VolumeMaskRadiiWizard._getListProvider(self, _objs)

    def show(self, form):
        params = self._getParameters(form.protocol)
        _value = params['value']
        _label = params['label']
        VolumeMaskRadiiWizard.show(self, form, _value, _label, UNIT_PIXEL)

class XmippVolumeRadiiProjMWizard(XmippVolumeRadiiWizard):
    _targets = [(XmippProtProjMatch, ['innerRadius', 'outerRadius'])]

    def _getParameters(self, protocol):

        label, value = self._getInputProtocol(self._targets, protocol)
        # Convert values to integer (From NumericListParam they come as string)
        value = [int(val) for val in value]

        protParams = {}
        protParams['input']= protocol.input3DReferences
        protParams['label']= label
        protParams['value']= value
        return protParams


#===============================================================================
#  FILTERS
#===============================================================================

class XmippFilterParticlesWizard(FilterParticlesWizard):
    _targets = [(XmippProtFilterParticles, ['lowFreqA','lowFreqDig',
                                            'highFreqA','highFreqDig',
                                            'freqDecayA','freqDecayDig'])]

    def _getParameters(self, protocol):

        protParams = {}

        if protocol.freqInAngstrom:
            labels = ['lowFreqA', 'highFreqA', 'freqDecayA']
            protParams['unit'] = UNIT_ANGSTROM
        else:
            labels = ['lowFreqDig', 'highFreqDig', 'freqDecayDig']
            protParams['unit'] = UNIT_PIXEL

        values = [protocol.getAttributeValue(l) for l in labels]

        protParams['input'] = protocol.inputParticles
        protParams['label'] = labels
        protParams['value'] = values
        protParams['mode'] = protocol.filterModeFourier.get()
        #protParams['space'] = protocol.filterSpace.get()
        #protParams['freqInAngstrom'] = protocol.freqInAngstrom.get()
        return protParams

    def _getProvider(self, protocol):
        _objs = self._getParameters(protocol)['input']
        return FilterParticlesWizard._getListProvider(self, _objs)

    def show(self, form):
        params = self._getParameters(form.protocol)
        _value = params['value']
        _label = params['label']
        _mode = params['mode']
        _unit = params['unit']
        FilterParticlesWizard.show(self, form, _value, _label, _mode, _unit)


class XmippFilterVolumesWizard(FilterVolumesWizard):
    _targets = [(XmippProtFilterVolumes, ['lowFreqA','lowFreqDig',
                                          'highFreqA','highFreqDig',
                                          'freqDecayA','freqDecayDig'])]

    def _getParameters(self, protocol):

        protParams = {}

        if protocol.freqInAngstrom:
            labels = ['lowFreqA', 'highFreqA', 'freqDecayA']
            protParams['unit'] = UNIT_ANGSTROM
        else:
            labels = ['lowFreqDig', 'highFreqDig', 'freqDecayDig']
            protParams['unit'] = UNIT_PIXEL

        values = [protocol.getAttributeValue(l) for l in labels]

        protParams['input']= protocol.inputVolumes
        protParams['label']= labels
        protParams['value']= values
        protParams['mode'] = protocol.filterModeFourier.get()
        #protParams['mode'] = protocol.filterSpace.get()
        #protParams['freqInAngstrom'] = protocol.freqInAngstrom.get()
        return protParams

    def _getProvider(self, protocol):
        _objs = self._getParameters(protocol)['input']
        return FilterVolumesWizard._getListProvider(self, _objs)

    def show(self, form):
        params = self._getParameters(form.protocol)
        _value = params['value']
        _label = params['label']
        _mode = params['mode']
        _unit = params['unit']
        FilterVolumesWizard.show(self, form, _value, _label, _mode, _unit)


class XmippGaussianParticlesWizard(GaussianParticlesWizard):
    _targets = [(XmippProtFilterParticles, ['freqSigma'])]

    def _getParameters(self, protocol):

        label, value = self._getInputProtocol(self._targets, protocol)

        protParams = {}
        protParams['input']= protocol.inputParticles
        protParams['label']= label
        protParams['value']= value
        return protParams

    def _getProvider(self, protocol):
        _objs = self._getParameters(protocol)['input']
        return GaussianParticlesWizard._getListProvider(self, _objs)

    def show(self, form):
        params = self._getParameters(form.protocol)
        _value = params['value']
        _label = params['label']
        GaussianParticlesWizard.show(self, form, _value, _label, UNIT_PIXEL_FOURIER)


class XmippGaussianVolumesWizard(GaussianVolumesWizard):
    _targets = [(XmippProtFilterVolumes, ['freqSigma'])]

    def _getParameters(self, protocol):

        label, value = self._getInputProtocol(self._targets, protocol)

        protParams = {}
        protParams['input']= protocol.inputVolumes
        protParams['label']= label
        protParams['value']= value
        return protParams

    def _getProvider(self, protocol):
        _objs = self._getParameters(protocol)['input']
        return GaussianVolumesWizard._getListProvider(self, _objs)

    def show(self, form):
        params = self._getParameters(form.protocol)
        _value = params['value']
        _label = params['label']
        GaussianVolumesWizard.show(self, form, _value, _label, UNIT_PIXEL_FOURIER)



