# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Jose Gutierrez (jose.gutierrez@cnb.csic.es)
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
# *  e-mail address 'jmdelarosa@cnb.csic.es'
# *
# **************************************************************************
"""
This module implement some wizards
"""

import os
import Tkinter as tk
import ttk

import xmipp

from pyworkflow.em.constants import *
from constants import *

from pyworkflow.em import SetOfImages, SetOfMicrographs, Volume, ProtCTFMicrographs
from protocol_projmatch import XmippProtProjMatch 
from protocol_preprocess_micrographs import XmippProtPreprocessMicrographs
from protocol_filter import XmippProtFilterParticles, XmippProtFilterVolumes
from protocol_mask import XmippProtMaskParticles, XmippProtMaskVolumes

from pyworkflow.em.wizard import * 


class XmippDownsampleWizard(downsampleWizard):
    _targets = [(XmippProtPreprocessMicrographs, ['downFactor'])]
    
class XmippCTFWizard(ctfWizard):
    _targets = [(ProtCTFMicrographs, ['lowRes', 'highRes'])]
            
# class XmippMaskRadiusWizard(maskRadiusWizard):
#     pass

class XmippParticleMaskRadiusWizard(particleMaskRadiusWizard):
    _targets = [(XmippProtMaskParticles, ['radius'])]
    
    def _getProvider(self, protocol):
        _objs = protocol.inputParticles.get()
        return particleMaskRadiusWizard._getProvider(self, protocol, _objs)
    
    def show(self, form):
        _value = form.protocol.radius.get()
        _label = "radius"
        particleMaskRadiusWizard.show(self, form, _value, _label, UNIT_PIXEL)
              
    
class XmippVolumeMaskRadiusWizard(volumeMaskRadiusWizard):
    _targets = [(XmippProtMaskVolumes, ['radius'])]
      
    def _getProvider(self, protocol):
        _objs = protocol.inputVolumes.get()    
        return volumeMaskRadiusWizard._getProvider(self, protocol, _objs)
    
    def show(self, form):
        _value = form.protocol.radius.get()
        _label = "radius"
        volumeMaskRadiusWizard.show(self, form, _value, _label, UNIT_PIXEL)
 
class XmippParticleMaskRadiiWizard(particlesMaskRadiiWizard):
    _targets = [(XmippProtMaskParticles, ['innerRadius', 'outerRadius'])]
    
    def _getProvider(self, protocol):
        _objs = protocol.inputParticles.get()
        return particlesMaskRadiiWizard._getProvider(self, protocol, _objs)
    
    def show(self, form):
        _value = [form.protocol.innerRadius.get(), form.protocol.outerRadius.get()]
        _label = ["innerRadius", "outerRadius"]
        particlesMaskRadiiWizard.show(self, form, _value, _label, UNIT_PIXEL)


class XmippVolumeRadiiWizard(volumeMaskRadiiWizard):
    _targets = [(XmippProtMaskVolumes, ['innerRadius', 'outerRadius'])]
 
    def _getProvider(self, protocol):
        _objs = protocol.inputVolumes.get()    
        return volumeMaskRadiiWizard._getProvider(self, protocol, _objs)
    
    def show(self, form):
        _value = [form.protocol.innerRadius.get(), form.protocol.outerRadius.get()]
        _label = ["innerRadius", "outerRadius"]
        
        volumeMaskRadiiWizard.show(self, form, _value, _label, UNIT_PIXEL)

# class XmippFilterParticlesWizard(filterParticlesWizard):
#     pass
 
# class XmippBandpassParticleWizard(bandpassParticleWizard):    
#     _targets = [(XmippProtFilterParticles, ['lowFreq', 'highFreq', 'freqDecay'])]
#     
#     def show(self, form):
#         protocol = form.protocol
#         provider = self._getProvider(protocol)
# 
#         if provider is not None:
#             self.mode = protocol.fourierMode.get()
#             args = {'mode':  self.mode,                   
#                     'lowFreq': protocol.lowFreq.get(),
#                     'highFreq': protocol.highFreq.get(),
#                     'freqDecay': protocol.freqDecay.get(),
#                     'unit': UNIT_PIXEL_FOURIER
#                     }
#             if self.mode == FILTER_LOW_PASS:
#                 args['showLowFreq'] = False
#             elif self.mode == FILTER_HIGH_PASS:
#                 args['showHighFreq'] = False
#             elif self.mode == FILTER_LOW_PASS_NO_DECAY:
#                 args['showLowFreq'] = False
#                 args['showDecay'] = False
#             elif self.mode == FILTER_BAND_PASS:
#                 pass
#             else:
#                 print "Not Mode"
#                 
#             d = bandPassFilterDialog(form.root, provider, **args)
#             
#             if d.resultYes():
#                 form.setVar('lowFreq', d.getLowFreq())
#                 form.setVar('highFreq', d.getHighFreq())
#                 form.setVar('freqDecay', d.getFreqDecay())
#         else:
#             dialog.showWarning("Input particles", "Select particles first", form.root)  
# 
# 
# class XmippBandpassVolumeWizard(bandpassVolumesWizard):    
#     _targets = [(XmippProtFilterVolumes, ['lowFreq', 'highFreq', 'freqDecay'])]
#     
#     def show(self, form):
#         protocol = form.protocol
#         provider = self._getProvider(protocol)
# 
#         if provider is not None:
#             self.mode = protocol.fourierMode.get()
#             args = {'mode':  self.mode,                   
#                     'lowFreq': protocol.lowFreq.get(),
#                     'highFreq': protocol.highFreq.get(),
#                     'freqDecay': protocol.freqDecay.get(),
#                     'unit': UNIT_PIXEL_FOURIER
#                     }
#             if self.mode == FILTER_LOW_PASS:
#                 args['showLowFreq'] = False
#             elif self.mode == FILTER_HIGH_PASS:
#                 args['showHighFreq'] = False
#             elif self.mode == FILTER_LOW_PASS_NO_DECAY:
#                 args['showLowFreq'] = False
#                 args['showDecay'] = False
#             elif self.mode == FILTER_BAND_PASS:
#                 pass
#             else:
#                 print "Not Mode"
#                 
#             d = bandPassFilterDialog(form.root, provider, **args)
#             
#             if d.resultYes():
#                 form.setVar('lowFreq', d.getLowFreq())
#                 form.setVar('highFreq', d.getHighFreq())
#                 form.setVar('freqDecay', d.getFreqDecay())
#         else:
#             dialog.showWarning("Input volume(s)", "Select volume(s) first", form.root)  


    
# class XmippGaussianWizard(gaussianWizard):
#     _targets = [(XmippProtFilter, ['freqSigma'])]
#     
#     def show(self, form):
#         protocol = form.protocol
#         provider = self._getProvider(protocol)
# 
#         if provider is not None:
#             d = gaussianFilterDialog(form.root, provider, freqSigma=protocol.freqSigma.get())
#             if d.resultYes():
#                 form.setVar('freqSigma', d.getFreqSigma())
#         else:
#             dialog.showWarning("Input particles", "Select particles first", form.root)  

    
  
